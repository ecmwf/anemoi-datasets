# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import itertools

from anemoi.datasets.dates import Dates
from anemoi.datasets.dates import no_time_zone


class Groups:
    """>>> list(Groups(group_by="daily", start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=12))[0]
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 12, 0)]

    >>> list(Groups(group_by="daily", start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=12))[1]
    [datetime.datetime(2023, 1, 2, 0, 0), datetime.datetime(2023, 1, 2, 12, 0)]

    >>> g = Groups(group_by=3, start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=24)
    >>> len(list(g))
    2
    >>> len(list(g)[0])
    3
    >>> len(list(g)[1])
    2
    >>> g = Groups(group_by=3,
    ...            start="2023-01-01 00:00",
    ...            end="2023-01-05 00:00",
    ...            frequency=24,
    ...            missing=["2023-01-02 00:00"])
    >>> len(list(g))
    2
    >>> len(list(g)[0])
    2
    >>> len(list(g)[1])
    2
    """

    def __init__(self, **kwargs):
        group_by = kwargs.pop("group_by")
        self.dates = Dates.from_config(**kwargs)
        self.grouper = Grouper.from_config(group_by)
        self.filter = Filter(self.dates.missing)

    def __iter__(self):
        for dates in self.grouper(self.dates):
            dates = self.filter(dates)
            if not dates:
                continue
            yield dates

    def __len__(self):
        count = 0
        for dates in self.grouper(self.dates):
            dates = self.filter(dates)
            if not dates:
                continue
            count += 1
        return count

    def __repr__(self):
        return f"{self.__class__.__name__}(dates={len(self)})"


class Filter:
    def __init__(self, missing):
        self.missing = [no_time_zone(m) for m in missing]

    def __call__(self, dates):
        return [d for d in dates if d not in self.missing]


class Grouper:
    @classmethod
    def from_config(cls, group_by):
        if isinstance(group_by, int) and group_by > 0:
            return GrouperByFixedSize(group_by)
        if group_by is None:
            return GrouperOneGroup()
        key = {
            "monthly": lambda dt: (dt.year, dt.month),
            "daily": lambda dt: (dt.year, dt.month, dt.day),
            "weekly": lambda dt: (dt.weekday(),),
            "MMDD": lambda dt: (dt.month, dt.day),
        }[group_by]
        return GrouperByKey(key)


class GrouperOneGroup(Grouper):
    def __call__(self, dates):
        yield dates.values


class GrouperByKey(Grouper):
    def __init__(self, key):
        self.key = key

    def __call__(self, dates):
        for _, g in itertools.groupby(dates, key=self.key):
            yield list(g)


class GrouperByFixedSize(Grouper):
    def __init__(self, size):
        self.size = size

    def __call__(self, dates):
        batch = []
        for d in dates:
            batch.append(d)
            if len(batch) == self.size:
                yield batch
                batch = []
        if batch:
            yield batch
