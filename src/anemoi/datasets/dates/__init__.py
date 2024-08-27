# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import warnings

# from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.dates import print_dates


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
                start += datetime.timedelta(hours=step)
            return

    yield as_datetime(x)


class Dates:
    """Base class for date generation.

    >>> Dates.from_config(**{"start": "2023-01-01 00:00", "end": "2023-01-02 00:00", "frequency": "1d"}).values
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)]

    >>> Dates.from_config(**{"start": "2023-01-01 00:00", "end": "2023-01-03 00:00", "frequency": "18h"}).values
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 18, 0), datetime.datetime(2023, 1, 2, 12, 0)]

    >>> Dates.from_config(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=6).as_dict()
    {'start': '2023-01-01T00:00:00', 'end': '2023-01-02T00:00:00', 'frequency': '6h'}

    >>> len(Dates.from_config(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=12))
    3
    >>> len(Dates.from_config(start="2023-01-01 00:00",
    ...                   end="2023-01-02 00:00",
    ...                   frequency=12,
    ...                   missing=["2023-01-01 12:00"]))
    3
    >>> len(Dates.from_config(start="2023-01-01 00:00",
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
            warnings.warn(f"Missing dates {self.missing} not in list.")

    @classmethod
    def from_config(cls, **kwargs):
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


class ValuesDates(Dates):
    def __init__(self, values, **kwargs):
        self.values = sorted([as_datetime(_) for _ in values])
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values[0]}..{self.values[-1]})"

    def as_dict(self):
        return {"values": self.values[0]}


class StartEndDates(Dates):
    def __init__(self, start, end, frequency=1, months=None, **kwargs):
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

        # if end <= start:
        #     raise ValueError(f"End date {end} must be after start date {start}")

        increment = frequency

        self.start = start
        self.end = end
        self.frequency = frequency

        date = start
        self.values = []
        while date <= end:

            if months is not None:
                if date.month not in months:
                    date += increment
                    continue

            self.values.append(date)
            date += increment

        super().__init__(**kwargs)

    def as_dict(self):
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "frequency": f"{self.frequency}h",
        }


if __name__ == "__main__":
    print_dates([datetime.datetime(2023, 1, 1, 0, 0)])
    s = StartEndDates(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=1)
    print_dates(list(s))
