# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
from anemoi.utils.dates import frequency_to_string


def _to_numpy_timedelta(td):
    if isinstance(td, np.timedelta64):
        assert td.dtype == "timedelta64[s]", f"expecting np.timedelta64[s], got {td.dtype}"
        return td
    return np.timedelta64(int(td.total_seconds()), "s")


def window_from_str(txt):
    """Parses a window string of the form '(-6h, 0h]' and returns a WindowsSpec object."""
    if txt.startswith("["):
        include_start = True
    elif txt.startswith("("):
        include_start = False
    else:
        raise ValueError(f"Invalid window {txt}, must start with '(' or '['")
    txt = txt[1:]

    if txt.endswith("]"):
        include_end = True
    elif txt.endswith(")"):
        include_end = False
    else:
        raise ValueError(f"Invalid window {txt}, must end with ')' or ']'")
    txt = txt[:-1]

    txt = txt.strip()
    if ";" in txt:
        txt = txt.replace(";", ",")
    lst = txt.split(",")
    if len(lst) != 2:
        raise ValueError(
            f"Invalid window {txt}, must be of the form '(start, end)' or '[start, end]' or '[start, end)' or '(start, end]'"
        )
    start, end = lst
    start = start.strip()
    end = end.strip()

    def _to_timedelta(t):
        # This part should go into utils
        from anemoi.utils.dates import as_timedelta

        if t.startswith(" ") or t.endswith(" "):
            t = t.strip()
        if t.startswith("-"):
            return -as_timedelta(t[1:])
        if t.startswith("+"):
            return as_timedelta(t[1:])
        # end of : This part should go into utils
        return as_timedelta(t)

    start = _to_timedelta(start)
    end = _to_timedelta(end)
    return WindowsSpec(
        start=start,
        end=end,
        include_start=include_start,
        include_end=include_end,
    )


class AbsoluteWindow:
    # not used but expected to be useful when building datasets. And used in tests
    def __init__(self, start, end, include_start=True, include_end=True):
        assert isinstance(start, datetime.datetime), f"start must be a datetime.datetime, got {type(start)}"
        assert isinstance(end, datetime.datetime), f"end must be a datetime.datetime, got {type(end)}"
        assert isinstance(include_start, bool), f"include_start must be a bool, got {type(include_start)}"
        assert isinstance(include_end, bool), f"include_end must be a bool, got {type(include_end)}"
        if start >= end:
            raise ValueError(f"start {start} must be less than end {end}")
        self.start = start
        self.end = end
        self.include_start = include_start
        self.include_end = include_end

    def __repr__(self):
        return f"{'[' if self.include_start else '('}{self.start.isoformat()},{self.end.isoformat()}{']' if self.include_end else ')'}"


class WindowsSpec:
    # A window specified by relative timedeltas, such as (-6h, 0h]
    #
    # the term "WindowSpec" is used here to avoid confusion between
    #       - a relative window, such as (-6h, 0h] which this class represents (WindowsSpec)
    #       - an actual time interval, such as [2023-01-01 00:00, 2023-01-01 06:00] which is an (AbsoluteWindow)
    #
    # but is is more confusing, it should be renamed as Window.

    def __init__(self, *, start, end, include_start=False, include_end=True):
        assert isinstance(start, (str, datetime.timedelta)), f"start must be a str or timedelta, got {type(start)}"
        assert isinstance(end, (str, datetime.timedelta)), f"end must be a str or timedelta, got {type(end)}"
        assert isinstance(include_start, bool), f"include_start must be a bool, got {type(include_start)}"
        assert isinstance(include_end, bool), f"include_end must be a bool, got {type(include_end)}"
        assert include_start in (True, False), f"Invalid include_start {include_start}"  # None is not allowed
        assert include_end in (True, False), f"Invalid include_end {include_end}"  # None is not allowed
        if start >= end:
            raise ValueError(f"start {start} must be less than end {end}")
        self.start = start
        self.end = end
        self.include_start = include_start
        self.include_end = include_end

        self._start_np = _to_numpy_timedelta(start)
        self._end_np = _to_numpy_timedelta(end)

    def to_absolute_window(self, date):
        """Convert the window to an absolute window based on a date."""
        # not used but expected to be useful when building datasets. And used in tests
        assert isinstance(date, datetime.datetime), f"date must be a datetime.datetime, got {type(date)}"
        start = date + self.start
        end = date + self.end
        return AbsoluteWindow(start=start, end=end, include_start=self.include_start, include_end=self.include_end)

    def __repr__(self):
        first = "[" if self.include_start else "("
        last = "]" if self.include_end else ")"

        def _frequency_to_string(t):
            if t < datetime.timedelta(0):
                return f"-{frequency_to_string(-t)}"
            elif t == datetime.timedelta(0):
                return "0"
            return frequency_to_string(t)

        return f"{first}{_frequency_to_string(self.start)},{_frequency_to_string(self.end)}{last}"

    def compute_mask(self, timedeltas):
        """Returns a boolean numpy array of the same shape as timedeltas."""

        assert timedeltas.dtype == "timedelta64[s]", f"expecting np.timedelta64[s], got {timedeltas.dtype}"
        if self.include_start:
            lower_mask = timedeltas >= self._start_np
        else:
            lower_mask = timedeltas > self._start_np

        if self.include_end:
            upper_mask = timedeltas <= self._end_np
        else:
            upper_mask = timedeltas < self._end_np

        return lower_mask & upper_mask

    def starts_before(self, my_dates, other_dates, other_window):
        # apply this window to my_dates[0] and the other_window to other_dates[0]
        # return True if this window starts before the other window

        assert my_dates.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {my_dates.dtype}"
        assert other_dates.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {other_dates.dtype}"
        assert isinstance(other_window, WindowsSpec), f"other_window must be a WindowsSpec, got {type(other_window)}"

        my_start = my_dates[0] + self._start_np
        other_start = other_dates[0] + other_window._start_np

        if my_start == other_start:
            return (not other_window.include_start) or self.include_start
        return my_start <= other_start

    def ends_after(self, my_dates, other_dates, other_window):
        # same as starts_before
        assert my_dates.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {my_dates.dtype}"
        assert other_dates.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {other_dates.dtype}"
        assert isinstance(other_window, WindowsSpec), f"other_window must be a WindowsSpec, got {type(other_window)}"

        my_end = my_dates[-1] + self._end_np
        other_end = other_dates[-1] + other_window._end_np

        if my_end == other_end:
            print(".", (not other_window.include_end) or self.include_end)
            return (not other_window.include_end) or self.include_end
        print(my_end >= other_end)
        return my_end >= other_end
