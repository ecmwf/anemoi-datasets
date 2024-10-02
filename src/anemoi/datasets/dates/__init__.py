# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import warnings

# from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import DateTimes
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.hindcasts import HindcastDatesTimes
from anemoi.utils.humanize import print_dates


def extend(x):

    if isinstance(x, (list, tuple)):
        for y in x:
            yield from extend(y)
        return

    if isinstance(x, str):
        if "/" in x:
            start, end, step = x.split("/")
            start = as_datetime(start)
            end = as_datetime(end)
            step = frequency_to_timedelta(step)
            while start <= end:
                yield start
                start += step
            return

    yield as_datetime(x)


class DatesProvider:
    """Base class for date generation.

    >>> DatesProvider.from_config(**{"start": "2023-01-01 00:00", "end": "2023-01-02 00:00", "frequency": "1d"}).values
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)]

    >>> DatesProvider.from_config(**{"start": "2023-01-01 00:00", "end": "2023-01-03 00:00", "frequency": "18h"}).values
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 18, 0), datetime.datetime(2023, 1, 2, 12, 0)]

    >>> DatesProvider.from_config(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=6).as_dict()
    {'start': '2023-01-01T00:00:00', 'end': '2023-01-02T00:00:00', 'frequency': '6h'}

    >>> len(DatesProvider.from_config(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=12))
    3
    >>> len(DatesProvider.from_config(start="2023-01-01 00:00",
    ...                   end="2023-01-02 00:00",
    ...                   frequency=12,
    ...                   missing=["2023-01-01 12:00"]))
    3
    >>> len(DatesProvider.from_config(start="2023-01-01 00:00",
    ...                   end="2023-01-02 00:00",
    ...                   frequency=12,
    ...                   missing=["2099-01-01 12:00"]))
    3
    """

    def __init__(self, missing=None):
        if not missing:
            missing = []
        self.missing = list(extend(missing))
        if set(self.missing) - set(self.values):
            diff = set(self.missing) - set(self.values)
            warnings.warn(f"Missing dates {len(diff)=} not in list.")

    @classmethod
    def from_config(cls, **kwargs):

        if kwargs.pop("hindcasts", False):
            return HindcastsDates(**kwargs)

        if "values" in kwargs:
            return ValuesDates(**kwargs)

        return StartEndDates(**kwargs)

    def __iter__(self):
        yield from self.values

    def __getitem__(self, i):
        return self.values[i]

    def __len__(self):
        return len(self.values)

    @property
    def summary(self):
        return f"📅 {self.values[0]} ... {self.values[-1]}"


class ValuesDates(DatesProvider):
    def __init__(self, values, **kwargs):
        self.values = sorted([as_datetime(_) for _ in values])
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values[0]}..{self.values[-1]})"

    def as_dict(self):
        return {"values": self.values[0]}


class StartEndDates(DatesProvider):
    def __init__(self, start, end, frequency=1, **kwargs):

        frequency = frequency_to_timedelta(frequency)
        assert isinstance(frequency, datetime.timedelta), frequency

        def _(x):
            if isinstance(x, str):
                return datetime.datetime.fromisoformat(x)
            return x

        start = _(start)
        end = _(end)

        if isinstance(start, datetime.date) and not isinstance(start, datetime.datetime):
            start = datetime.datetime(start.year, start.month, start.day)

        if isinstance(end, datetime.date) and not isinstance(end, datetime.datetime):
            end = datetime.datetime(end.year, end.month, end.day)

        start = as_datetime(start)
        end = as_datetime(end)

        self.start = start
        self.end = end
        self.frequency = frequency

        missing = kwargs.pop("missing", [])

        self.values = list(DateTimes(start, end, increment=frequency, **kwargs))
        self.kwargs = kwargs

        super().__init__(missing=missing)

    def as_dict(self):
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "frequency": frequency_to_string(self.frequency),
        }.update(self.kwargs)


class Hindcast:

    def __init__(self, date, refdate, hdate, step):
        self.date = date
        self.refdate = refdate
        self.hdate = hdate
        self.step = step


class HindcastsDates(DatesProvider):
    def __init__(self, start, end, steps=[0], years=20, **kwargs):

        if not isinstance(start, list):
            start = [start]
            end = [end]

        reference_dates = []
        for s, e in zip(start, end):
            reference_dates.extend(list(DateTimes(s, e, increment=24, **kwargs)))
        # reference_dates = list(DateTimes(start, end, increment=24, **kwargs))
        dates = []

        seen = {}

        for hdate, refdate in HindcastDatesTimes(reference_dates=reference_dates, years=years):
            assert refdate - hdate >= datetime.timedelta(days=365), (refdate - hdate, refdate, hdate)
            for step in steps:

                date = hdate + datetime.timedelta(hours=step)

                if date in seen:
                    raise ValueError(f"Duplicate date {date}={hdate}+{step} for {refdate} and {seen[date]}")

                seen[date] = Hindcast(date, refdate, hdate, step)

                assert refdate - date > datetime.timedelta(days=360), (refdate - date, refdate, date, hdate, step)

                dates.append(date)

        dates = sorted(dates)

        mindelta = None
        for a, b in zip(dates, dates[1:]):
            delta = b - a
            assert isinstance(delta, datetime.timedelta), delta
            if mindelta is None:
                mindelta = delta
            else:
                mindelta = min(mindelta, delta)

        self.frequency = mindelta
        assert mindelta.total_seconds() > 0, mindelta

        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥", dates[0], dates[-1], mindelta)

        # Use all values between start and end by frequency, and set the ones that are missing
        self.values = []
        missing = []
        date = dates[0]
        last = date
        print("------", date, dates[-1])
        dateset = set(dates)
        while date <= dates[-1]:
            self.values.append(date)
            if date not in dateset:
                missing.append(date)
                seen[date] = seen[last]
            else:
                last = date
            date = date + mindelta

        self.mapping = seen

        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥", self.values[0], self.values[-1], mindelta)
        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥", f"{len(self.values)=} - {len(missing)=}")

        super().__init__(missing=missing)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values[0]}..{self.values[-1]})"

    def as_dict(self):
        return {"hindcasts": self.hindcasts}


if __name__ == "__main__":
    print_dates([datetime.datetime(2023, 1, 1, 0, 0)])
    s = StartEndDates(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=1)
    print_dates(list(s))
