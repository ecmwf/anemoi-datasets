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
from .grids import GridsBase
from .indexing import apply_index_to_slices_changes
from .indexing import index_to_slices
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Augment(GridsBase):

    def __init__(self, datasets, axis=3, what="variables", method="nearest"):

        super().__init__(datasets, axis=axis)
        assert len(datasets) == 2, "Augment requires two datasets"
        assert axis == 3, "Augment requires axis=3"

        # We had the variables of dataset[1] to dataset[0]
        # interpoated on the grid of dataset[0]

        self.target = self.datasets[0]
        self.source = self.datasets[1]

        self._variables = []

        # Keep the same order as the original dataset
        for v in self.source.variables:
            if v not in self.target.variables:
                self._variables.append(v)

        if not self._variables:
            raise ValueError("Augment: no missing variables")

        self._nearest_grid_points = nearest_grid_points(
            self.source.latitudes, self.source.longitudes, self.target.latitudes, self.target.longitudes
        )

    @property
    def variables(self):
        return self._variables

    @property
    def shape(self):
        return self.target.shape

    @property
    def variables_metadata(self):
        return {k: v for k, v in self.source.variables_metadata.items() if k in self._variables}

    def check_same_variables(self, d1, d2):
        pass

    def check_compatibility(self, d1, d2):
        pass

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            index = (index, slice(None), slice(None), slice(None))
        return self._get_tuple(index)

    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)

        source_data = self.source[index[:3]]
        target_data = source_data[..., self._nearest_grid_points]

        result = target_data[..., index[3]]

        return apply_index_to_slices_changes(result, changes)

    def tree(self):
        """Generates a hierarchical tree structure for the `Cutout` instance and
        its associated datasets.

        Returns:
            Node: A `Node` object representing the `Cutout` instance as the root
            node, with each dataset in `self.datasets` represented as a child
            node.
        """
        return Node(self, [d.tree() for d in self.datasets])


def augment_factory(args, kwargs):
    if "ensemble" in kwargs:
        raise NotImplementedError("Cannot use both 'ensemble' and 'augment'")

    augment = kwargs.pop("augment")
    what = kwargs.pop("what", "variables")
    method = kwargs.pop("method", "nearest")

    if what != "variables":
        raise NotImplementedError(f"Augment what={what} not implemented")

    if method != "nearest":
        raise NotImplementedError(f"Augment method={method} not implemented")

    assert len(args) == 0
    assert isinstance(augment, (list, tuple)), "augment must be a list or tuple"

    datasets = [_open(e) for e in augment]
    # select is the same as variables
    datasets, kwargs = _auto_adjust(datasets, kwargs, exclude=["select"])

    return Augment(
        datasets,
        what=what,
        method=method,
    )._subset(**kwargs)
