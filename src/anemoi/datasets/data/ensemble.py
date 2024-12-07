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

OFFSETS = dict(number=1, numbers=1, member=0, members=0)


class Number(Forwards):
    def __init__(self, forward, **kwargs):
        super().__init__(forward)

        self.members = []
        for key, values in kwargs.items():
            if not isinstance(values, (list, tuple)):
                values = [values]
            self.members.extend([int(v) - OFFSETS[key] for v in values])

        self.members = sorted(set(self.members))
        for n in self.members:
            if not (0 <= n < forward.shape[2]):
                raise ValueError(f"Member {n} is out of range. `number(s)` is one-based, `member(s)` is zero-based.")

        self.mask = np.array([n in self.members for n in range(forward.shape[2])], dtype=bool)
        self._shape, _ = update_tuple(forward.shape, 2, len(self.members))

    @property
    def shape(self):
        return self._shape

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
        return Node(self, [self.forward.tree()], numbers=[n + 1 for n in self.members])

    def metadata_specific(self):
        return {
            "numbers": [n + 1 for n in self.members],
        }


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
