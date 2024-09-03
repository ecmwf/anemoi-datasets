# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta

from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple
from .misc import _open

LOG = logging.getLogger(__name__)


class InterpolateTime(Forwards):

    def __init__(self, dataset, frequency):
        super().__init__(dataset)
        self._frequency = frequency_to_timedelta(frequency)

        mine = self._frequency.total_seconds()
        other = dataset.frequency.total_seconds()

        mine = int(mine)
        assert mine == self._frequency.total_seconds()

        other = int(other)
        assert other == dataset.frequency.total_seconds()

        if mine >= other:
            raise ValueError(f"Interpolate frequency must be higher {self._frequency} <= {dataset.frequency}")

        assert other % mine == 0, (other, mine)
        self.ratio = other // mine
        self.seconds = mine
        self.alphas = np.linspace(0, 1, self.ratio + 1)
        self.other_len = len(dataset)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 0, slice(None))
        result = self._get_slice(previous)
        return apply_index_to_slices_changes(result[index], changes)

    def _get_slice(self, s):
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        if n < 0:
            n += self._len

        if n == self._len - 1:
            return self.forward[-1]

        i1 = n // self.ratio
        x = n % self.ratio

        if x == 0:
            return self.forward[i1]

        alphas = self.alphas[x]
        return self.forward[i1] * (1 - alphas) + self.forward[i1 + 1] * alphas

    def __len__(self):
        return len(self.forward) * self.ratio

    @property
    def frequency(self):
        return self._frequency

    @cached_property
    def dates(self):
        result = []
        deltas = [np.timedelta64(self.seconds * i, "s") for i in range(self.ratio)]

        for d in self.forward.dates:
            for i in deltas[:-1]:
                result.append(d + i)
            result.append(d)
        return result

    @property
    def shape(self):
        return (len(self),) + self.forward.shape[1:]

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])

    @property
    def missing(self):
        raise NotImplementedError("missing not implemented for InterpolateTime")
        return self.forward.missing


def interpolate_factory(args, kwargs):
    assert len(args) == 0

    axis = kwargs.pop("axis", "time")
    assert axis in ("time",)

    dataset = kwargs.pop("interpolate")
    frequency = kwargs.pop("frequency")

    dataset = _open(dataset)

    return InterpolateTime(
        dataset,
        frequency=frequency,
    )._subset(**kwargs)
