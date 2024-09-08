# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np

from anemoi.datasets.create.utils import to_datetime
from anemoi.datasets.data import MissingDateError

from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import expand_list_indexing
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class MissingDates(Forwards):
    # TODO: Use that class instead of ZarrMissing

    def __init__(self, dataset, missing_dates):
        super().__init__(dataset)
        self.missing_dates = []

        self._missing = set()

        other = []
        for date in missing_dates:
            if isinstance(date, int):
                self._missing.add(date)
                self.missing_dates.append(dataset.dates[date])
            else:
                date = to_datetime(date)
                other.append(date)

        if other:
            for i, date in enumerate(dataset.dates):
                if date in other:
                    self._missing.add(i)
                    self.missing_dates.append(date)

        n = self.forward._len
        self._missing = set(i for i in self._missing if 0 <= i < n)
        self.missing_dates = sorted(to_datetime(x) for x in self.missing_dates)

        assert len(self._missing), "No dates to force missing"

    @cached_property
    def missing(self):
        return self._missing.union(self.forward.missing)

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n):
        if isinstance(n, int):
            if n in self.missing:
                self._report_missing(n)
            return self.forward[n]

        if isinstance(n, slice):
            common = set(range(*n.indices(len(self)))) & self.missing
            if common:
                self._report_missing(list(common)[0])
            return self.forward[n]

        if isinstance(n, tuple):
            first = n[0]
            if isinstance(first, int):
                if first in self.missing:
                    self._report_missing(first)
                return self.forward[n]

            if isinstance(first, slice):
                common = set(range(*first.indices(len(self)))) & self.missing
                if common:
                    self._report_missing(list(common)[0])
                return self.forward[n]

            if isinstance(first, (list, tuple)):
                common = set(first) & self.missing
                if common:
                    self._report_missing(list(common)[0])
                return self.forward[n]

        raise TypeError(f"Unsupported index {n} {type(n)}")

    def _report_missing(self, n):
        raise MissingDateError(f"Date {self.forward.dates[n]} is missing (index={n})")

    @property
    def reason(self):
        return {"missing_dates": self.missing_dates}

    def tree(self):
        return Node(self, [self.forward.tree()], **self.reason)

    def subclass_metadata_specific(self):
        return {"missing_dates": self.missing_dates}


class SkipMissingDates(Forwards):

    def __init__(self, dataset, expected_access):
        super().__init__(dataset)

        # if isinstance(expected_access, (tuple, list)):
        #     expected_access = slice(*expected_access)

        if isinstance(expected_access, int):
            expected_access = slice(0, expected_access)

        assert isinstance(expected_access, slice), f"Expected access must be a slice, got {expected_access}"

        expected_access = slice(*expected_access.indices(dataset._len))
        missing = dataset.missing.copy()

        size = (expected_access.stop - expected_access.start) // expected_access.step
        indices = []

        for i in range(dataset._len):
            s = slice(expected_access.start + i, expected_access.stop + i, expected_access.step)
            p = set(range(*s.indices(dataset._len)))
            if p.intersection(missing):
                continue

            if len(p) != size:
                continue

            indices.append(tuple(sorted(p)))

        self.expected_access = expected_access
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    @property
    def start_date(self):
        return self.forward.start_date

    @property
    def end_date(self):
        return self.forward.end_date

    @property
    def dates(self):
        raise NotImplementedError("SkipMissingDates.dates")

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):

        def _get_one(n):
            result = []
            for i in self.indices[n]:
                s, _ = update_tuple(index, 0, i)
                result.append(self.forward[s])

            return tuple(result)

        first = index[0]
        if isinstance(first, int):
            return _get_one(first)

        assert isinstance(first, slice), f"SkipMissingDates._get_tuple {index}"

        values = [_get_one(i) for i in range(*first.indices(self._len))]

        result = [_ for _ in zip(*values)]
        return tuple(np.stack(_) for _ in result)

    @debug_indexing
    def _get_slice(self, s):
        values = [self[i] for i in range(*s.indices(self._len))]
        result = [_ for _ in zip(*values)]
        return tuple(np.stack(_) for _ in result)

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        return tuple(self.forward[i] for i in self.indices[n])

    @property
    def frequency(self):
        return self.forward.frequency

    def tree(self):
        return Node(self, [self.forward.tree()], expected_access=self.expected_access)

    def subclass_metadata_specific(self):
        return {"expected_access": self.expected_access}


class MissingDataset(Forwards):

    def __init__(self, dataset, start, end):
        super().__init__(dataset)
        self.start = start
        self.end = end

        dates = []
        date = start
        while date <= end:
            dates.append(date)
            date += dataset.frequency

        self._dates = np.array(dates, dtype="datetime64")
        self._missing = set(range(len(dates)))

    def __len__(self):
        return len(self._dates)

    @property
    def dates(self):
        return self._dates

    @property
    def missing(self):
        return self._missing

    def __getitem__(self, n):
        raise MissingDateError(f"Date {self.dates[n]} is missing (index={n})")

    def tree(self):
        return Node(self, [self.forward.tree()], start=self.start, end=self.end)

    def subclass_metadata_specific(self):
        return {"start": self.start, "end": self.end}
