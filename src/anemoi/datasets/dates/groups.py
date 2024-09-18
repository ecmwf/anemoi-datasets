# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import itertools
from functools import cached_property

from anemoi.datasets.create.input import shorten
from anemoi.datasets.dates import Dates
from anemoi.datasets.dates import as_datetime


class GroupOfDates:
    def __init__(self, dates, provider):
        assert isinstance(provider, Dates), type(provider)
        assert isinstance(dates, list)

        self._dates = dates
        self._provider = provider

    def __len__(self):
        return len(self._dates)

    def __iter__(self):
        return iter(self._dates)


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
        self._dates = Dates.from_config(**kwargs)
        self._grouper = Grouper.from_config(group_by)
        self._filter = Filter(self._dates.missing)

    @property
    def provider(self):
        return self._dates

    def __iter__(self):
        for go in self._grouper(self._dates):
            dates = self._filter(go._dates)
            if not dates:
                continue
            yield GroupOfDates(dates, go._provider)

    def __len__(self):
        return self._len

    @cached_property
    def _len(self):
        n = 0
        for go in self._grouper(self._dates):
            dates = self._filter(go._dates)
            if not dates:
                continue
            n += 1
        return n

    def __repr__(self):
        return f"{self.__class__.__name__}(dates={len(self)},{shorten(self._dates)})"

    def describe(self):
        return self.dates.summary

    def one_date(self):
        go = next(iter(self))
        return GroupOfDates([go._dates[0]], go._provider)


class Filter:
    def __init__(self, missing):
        self.missing = [as_datetime(m) for m in missing]

    def __call__(self, dates):
        return [d for d in dates if d not in self.missing]


class Grouper:
    @classmethod
    def from_config(cls, group_by):

        if isinstance(group_by, int) and group_by > 0:
            return GrouperByFixedSize(group_by)

        if group_by is None:
            return GrouperOneGroup()

        if group_by == "reference_date":
            return ReferenceDateGroup()

        key = {
            "monthly": lambda dt: (dt.year, dt.month),
            "daily": lambda dt: (dt.year, dt.month, dt.day),
            "weekly": lambda dt: (dt.weekday(),),
            "MMDD": lambda dt: (dt.month, dt.day),
        }[group_by]
        return GrouperByKey(key)


class ReferenceDateGroup(Grouper):
    def __call__(self, dates):
        assert isinstance(dates, Dates), type(dates)

        mapping = dates.mapping

        def same_refdate(dt):
            refdate, _ = mapping[dt]
            return refdate

        for _, g in itertools.groupby(sorted(dates, key=same_refdate), key=same_refdate):
            yield GroupOfDates(list(g), dates)


class GrouperOneGroup(Grouper):
    def __call__(self, dates):
        assert isinstance(dates, Dates), type(dates)

        yield GroupOfDates(dates.values, dates)


class GrouperByKey(Grouper):
    def __init__(self, key):
        self.key = key

    def __call__(self, dates):
        for _, g in itertools.groupby(sorted(dates, key=self.key), key=self.key):
            yield GroupOfDates(list(g), dates)


class GrouperByFixedSize(Grouper):
    def __init__(self, size):
        self.size = size

    def __call__(self, dates):
        batch = []

        for d in dates:
            batch.append(d)
            if len(batch) == self.size:
                yield GroupOfDates(batch, dates)
                batch = []

        if batch:
            yield GroupOfDates(batch, dates)
