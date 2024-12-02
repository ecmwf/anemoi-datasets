# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..grids import nearest_grid_points
from .debug import Node
from .forwards import Combined
from .indexing import apply_index_to_slices_changes
from .indexing import index_to_slices
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Complement(Combined):

    def __init__(self, target, source, what="variables", interpolation="nearest"):
        super().__init__([target, source])

        # We had the variables of dataset[1] to dataset[0]
        # interpoated on the grid of dataset[0]

        self.target = target
        self.source = source

        self._variables = []

        # Keep the same order as the original dataset
        for v in self.source.variables:
            if v not in self.target.variables:
                self._variables.append(v)

        if not self._variables:
            raise ValueError("Augment: no missing variables")

    @property
    def variables(self):
        return self._variables

    @property
    def shape(self):
        shape = self.target.shape
        return (shape[0], len(self._variables)) + shape[2:]

    @property
    def variables_metadata(self):
        return {k: v for k, v in self.source.variables_metadata.items() if k in self._variables}

    def check_same_variables(self, d1, d2):
        pass

    def tree(self):
        """Generates a hierarchical tree structure for the `Cutout` instance and
        its associated datasets.

        Returns:
            Node: A `Node` object representing the `Cutout` instance as the root
            node, with each dataset in `self.datasets` represented as a child
            node.
        """
        return Node(self, [d.tree() for d in (self.target, self.source)])

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            index = (index, slice(None), slice(None), slice(None))
        return self._get_tuple(index)


class ComplementNone(Complement):

    def __init__(self, target, source):
        super().__init__(target, source)

    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        result = self.source[index]
        return apply_index_to_slices_changes(result, changes)


class ComplementNearest(Complement):

    def __init__(self, target, source):
        super().__init__(target, source)

        self._nearest_grid_points = nearest_grid_points(
            self.source.latitudes,
            self.source.longitudes,
            self.target.latitudes,
            self.target.longitudes,
        )

    def check_compatibility(self, d1, d2):
        pass

    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)

        source_data = self.source[index[:3]]
        target_data = source_data[..., self._nearest_grid_points]

        result = target_data[..., index[3]]

        return apply_index_to_slices_changes(result, changes)


def complement_factory(args, kwargs):
    assert len(args) == 0, args

    source = kwargs.pop("source")
    target = kwargs.pop("complement")
    what = kwargs.pop("what", "variables")
    interpolation = kwargs.pop("interpolation", "none")

    if what != "variables":
        raise NotImplementedError(f"Complement what={what} not implemented")

    if interpolation not in ("none", "nearest"):
        raise NotImplementedError(f"Complement method={interpolation} not implemented")

    source = _open(source)
    target = _open(target)
    # `select` is the same as `variables`
    (source, target), kwargs = _auto_adjust([source, target], kwargs, exclude=["select"])

    Class = {
        None: ComplementNone,
        "none": ComplementNone,
        "nearest": ComplementNearest,
    }[interpolation]

    complement = Class(target=target, source=source)._subset(**kwargs)

    # Will join the datasets along the variables axis
    return _open([target, complement])
