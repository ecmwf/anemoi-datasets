# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np

from anemoi.datasets.data import MissingDateError

from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class MissingDatesFill(Forwards):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._missing = set(dataset.missing)
        self._warnings = set()

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 0, slice(None))
        result = self._get_slice(previous)
        return apply_index_to_slices_changes(result[index], changes)

    def _get_slice(self, s):
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @property
    def missing(self):
        return set()

    @debug_indexing
    def __getitem__(self, n):

        try:
            return self.forward[n]
        except MissingDateError:
            pass

        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        if n < 0:
            n += self._len

        a = None
        i = n
        while a is None and i >= 0:
            if i in self._missing:
                i -= 1
            else:
                a = i

        len = self._len
        b = None
        i = n
        while b is None and n < len:
            if i in self._missing:
                i += 1
            else:
                b = i

        return self._fill_missing(n, a, b)


class MissingDatesClosest(MissingDatesFill):

    def __init__(self, dataset, closest):
        super().__init__(dataset)
        self.closest = closest
        self._closest = {}

    def _fill_missing(self, n, a, b):

        if n not in self._warnings:
            LOG.warning(f"Missing date at index {n} ({self.dates[n]})")
            if abs(n - a) == abs(b - n):
                if self.closest == "up":
                    u = b
                else:
                    u = a
            else:
                if abs(n - a) < abs(b - n):
                    u = a
                else:
                    u = b
            LOG.warning(f"Using closest date {u} ({self.dates[u]})")

            self._closest[n] = u
            self._warnings.add(n)

        return self.forward[self._closest[n]]

    def subclass_metadata_specific(self):
        return {"closest": self.closest}

    def tree(self):
        return Node(self, [self.forward.tree()], closest=self.closest)


class MissingDatesInterpolate(MissingDatesFill):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._alpha = {}

    def _fill_missing(self, n, a, b):
        if n not in self._warnings:
            LOG.warning(f"Missing date at index {n} ({self.dates[n]})")

            if a is None or b is None:
                raise MissingDateError(
                    f"Cannot interpolate at index {n} ({self.dates[n]}). Are the first or last date missing?"
                )

            assert a < n < b, (a, n, b)

            alpha = (n - a) / (b - a)
            assert 0 < alpha < 1, alpha

            LOG.warning(f"Interpolating between index {a} ({self.dates[a]}) and {b} ({self.dates[b]})")
            LOG.warning(f"Interpolation {1 - alpha:g} * ({self.dates[a]}) + {alpha:g} * ({self.dates[b]})")

            self._alpha[n] = alpha

            self._warnings.add(n)

        alpha = self._alpha[n]
        return self.forward[a] * (1 - alpha) + self.forward[b] * alpha

    def subclass_metadata_specific(self):
        return {}

    def tree(self):
        return Node(self, [self.forward.tree()])


def fill_missing_dates_factory(dataset, method, kwargs):
    if method == "closest":
        closest = kwargs.get("closest", "up")
        return MissingDatesClosest(dataset, closest=closest)

    if method == "interpolate":
        return MissingDatesInterpolate(dataset)

    raise ValueError(f"Invalid `fill_missing_dates` method '{method}'")
