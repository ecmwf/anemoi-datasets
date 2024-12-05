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

from .debug import Node
from .forwards import Forwards
from .forwards import GivenAxis
from .indexing import apply_index_to_slices_changes
from .indexing import index_to_slices
from .indexing import update_tuple
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Number(Forwards):
    def __init__(self, forward, numbers):
        super().__init__(forward)
        if not isinstance(numbers, (list, tuple)):
            numbers = [numbers]

        numbers = [int(n) for n in numbers]

        for n in numbers:
            assert 1 <= n <= forward.shape[2], "Invalid number. `number` is one-based"

        self.mask = np.array([n + 1 in numbers for n in range(forward.shape[2])], dtype=bool)

    @property
    def shape(self):
        shape, _ = update_tuple(self.forward.shape, 2, len(self.mask))
        return shape

    def __getitem__(self, index):
        if isinstance(index, int):
            result = self.forward[index]
            result = result[:, self.mask, :]
            return result

        if isinstance(index, slice):
            result = self.forward[index]
            result = result[:, :, self.mask, :]
            return result

        index, changes = index_to_slices(index, self.shape)
        result = self.forward[index]
        result = result[:, :, self.mask, :]
        return apply_index_to_slices_changes(result, changes)

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])


class Ensemble(GivenAxis):
    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])


def ensemble_factory(args, kwargs):
    if "grids" in kwargs:
        raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")

    ensemble = kwargs.pop("ensemble")
    axis = kwargs.pop("axis", 2)
    assert len(args) == 0
    assert isinstance(ensemble, (list, tuple))

    datasets = [_open(e) for e in ensemble]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Ensemble(datasets, axis=axis)._subset(**kwargs)
