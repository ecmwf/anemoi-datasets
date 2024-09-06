# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

from anemoi.datasets.create.utils import to_datetime
from anemoi.datasets.data import MissingDateError
from anemoi.datasets.data.subset import Subset

from .debug import debug_indexing
from .forwards import Forwards
from .indexing import expand_list_indexing
from .misc import _open

LOG = logging.getLogger(__name__)


class ForceMissing(Forwards):

    def __init__(self, dataset, dates):
        super().__init__(dataset)

        self._missing = set()

        other = []
        for date in dates:
            if isinstance(date, int):
                self._missing.add(date)
            else:
                other.append(to_datetime(date))

        if other:
            for i, date in enumerate(dataset.dates):
                if date in other:
                    self._missing.add(i)

        n = self.forward._len
        self._missing = set(i for i in self._missing if 0 <= i < n)

        assert len(self._missing), "No dates to force missing"

    @cached_property
    def missing(self):
        print("missing", self._missing, self.forward.missing)
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

    def subclass_metadata_specific(self):
        return {
            "missing": sorted(self.missing),
        }


class SkipMissing(Subset):

    def __init__(self, dataset, access):

        if isinstance(access, (tuple, list)):
            access = slice(*access)

        if isinstance(access, int):
            access = slice(0, access)

        access = slice(*access.indices(dataset._len))
        missing = dataset.missing.copy()

        width = access.stop - access.start

        skip = set()

        for i in missing:
            j = max(i - width, 0)
            k = min(i + width, dataset._len)
            for m in range(j, k):
                s = slice(access.start + m, access.stop + m, access.step)
                p = set(range(*s.indices(dataset._len)))
                if p.intersection(missing):
                    skip.add(m)

        missing |= skip

        indices = [i for i in range(dataset._len) if i not in missing]
        super().__init__(dataset, indices, reason=dict(access=access))

    @cached_property
    def dates(self):
        return self.forward.dates

    @property
    def frequency(self):
        return self.forward.frequency


def missing_factory(args, kwargs):
    assert len(args) == 0

    dataset = kwargs.pop("missing")
    dates = kwargs.pop("dates", None)

    dataset = _open(dataset)

    if dates:
        return ForceMissing(
            dataset,
            dates=dates,
        )._subset(**kwargs)

    access = kwargs.pop("access")

    return SkipMissing(dataset, access=access)._subset(**kwargs)
