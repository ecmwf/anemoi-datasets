# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import re
from functools import cached_property

import numpy as np
import zarr
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from .btree import ZarrBTree
from .caching import ChunksCache


class Window:
    def __init__(self, window: str):
        m = re.match(r"([\[\(])(.*),(.*)([\]\)])", window)
        if not m:
            raise ValueError(f"Window: invalid window string: {window}")
        self.before = frequency_to_timedelta(m.group(2))
        self.after = frequency_to_timedelta(m.group(3))
        self.exclude_before = m.group(1) == "("
        self.exclude_after = m.group(4) == ")"

    def __repr__(self):
        B = {True: ("(", ")"), False: ("[", "]")}
        return (
            f"{B[self.exclude_before][0]}{frequency_to_string(self.before)},"
            f"{frequency_to_string(self.after)}{B[self.exclude_after][1]}"
        )


class WindowView:

    def __init__(
        self,
        store,
        start_date=None,
        end_date=None,
        frequency=3,
        window="(-3,+0]",
        btree=None,
    ):
        self.store = store if isinstance(store, zarr.hierarchy.Group) else zarr.open(store, mode="r")
        self.btree = btree if btree is not None else ZarrBTree(self.store, mode="r")
        self.data = ChunksCache(self.store["data"])

        self.start_date = start_date if start_date is not None else self.actual_start_end_dates[0]
        self.end_date = end_date if end_date is not None else self.actual_start_end_dates[1]

        if self.start_date > self.end_date:
            raise ValueError(f"WindowView: {start_date=} must be less than or equal to {end_date=}")

        self.frequency = frequency_to_timedelta(frequency)
        self.window = window if isinstance(window, Window) else Window(window)

        self._len = (self.end_date - self.start_date) // self.frequency + 1

    def set_start(self, start: datetime.datetime) -> "WindowView":
        return self
        return WindowView(
            store=self.store,
            start_date=start,
            end_date=self.end_date,
            frequency=self.frequency,
            window=self.window,
            btree=self.btree,
        )

    def set_end(self, end: datetime.datetime) -> "WindowView":
        return self
        return WindowView(
            store=self.store,
            start_date=self.start_date,
            end_date=end,
            frequency=self.frequency,
            window=self.window,
            btree=self.btree,
        )

    def set_frequency(self, frequency: str | int | datetime.timedelta) -> "WindowView":
        return WindowView(
            store=self.store,
            start_date=self.start_date,
            end_date=self.end_date,
            frequency=frequency,
            window=self.window,
            btree=self.btree,
        )

    def set_window(self, window: str | Window) -> "WindowView":
        return WindowView(
            store=self.store,
            start_date=self.start_date,
            end_date=self.end_date,
            frequency=self.frequency,
            window=window,
            btree=self.btree,
        )

    @cached_property
    def actual_start_end_dates(self):
        start, end = self.btree.start_end()
        print("actual_start_end_dates", start, end)
        return datetime.datetime.fromtimestamp(start[0]), datetime.datetime.fromtimestamp(end[0])

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        assert isinstance(index, int)
        if index < 0:
            index = self._len - index

        if not 0 <= index < self._len:
            raise IndexError(f"Index {index} out of range (len={self._len})")

        start = self.start_date + index * self.frequency + self.window.before
        end = self.start_date + index * self.frequency + self.window.after

        start = round(start.timestamp())
        end = round(end.timestamp())

        first, last = self.btree.boundaries(start, end)

        if (first, last) == (None, None):
            shape = (0,) + self.data.shape[1:]
            return np.zeros(shape=shape, dtype=self.data.dtype)

        first_date, (start_idx, start_cnt) = first
        last_date, (end_idx, end_cnt) = last

        last_idx = end_idx + end_cnt

        assert first_date >= start
        assert last_date <= end

        if self.window.exclude_before and first_date == start:
            start_idx += start_cnt

        if self.window.exclude_after and last_date == end:
            last_idx -= end_cnt

        return self.data[start_idx:last_idx]

    def __repr__(self):
        return (
            f"WindowView(start_date={self.start_date}, end_date={self.end_date}, "
            f"frequency={self.frequency}, window={self.window})"
        )
