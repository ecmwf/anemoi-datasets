# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from numpy.typing import NDArray

from ..grids import nearest_grid_points
from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import Node
from .forwards import Combined
from .indexing import apply_index_to_slices_changes
from .indexing import index_to_slices
from .indexing import update_tuple
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Complement(Combined):

    def __init__(
        self,
        target: Dataset,
        source: Dataset,
        what: str = "variables",
        interpolation: str = "nearest",
    ) -> None:
        super().__init__([target, source])

        # We had the variables of dataset[1] to dataset[0]
        # interpoated on the grid of dataset[0]

        self._target: Dataset = target
        self._source: Dataset = source

        self._variables = []

        # Keep the same order as the original dataset
        for v in self._source.variables:
            if v not in self._target.variables:
                self._variables.append(v)

        if not self._variables:
            raise ValueError("Augment: no missing variables")

    @property
    def variables(self) -> List[str]:
        return self._variables

    @property
    def name_to_index(self) -> Dict[str, int]:
        return {v: i for i, v in enumerate(self.variables)}

    @property
    def shape(self) -> Shape:
        shape = self._target.shape
        return (shape[0], len(self._variables)) + shape[2:]

    @property
    def variables_metadata(self) -> dict:
        return {k: v for k, v in self._source.variables_metadata.items() if k in self._variables}

    def check_same_variables(self, d1: Dataset, d2: Dataset) -> None:
        pass

    @cached_property
    def missing(self) -> Set[int]:
        missing = self._source.missing.copy()
        missing = missing | self._target.missing
        return set(missing)

    def tree(self) -> Node:
        """Generates a hierarchical tree structure for the `Cutout` instance and
        its associated datasets.

        Returns:
            Node: A `Node` object representing the `Cutout` instance as the root
            node, with each dataset in `self.datasets` represented as a child
            node.
        """
        return Node(self, [d.tree() for d in (self._target, self._source)])

    def __getitem__(self, index: FullIndex) -> NDArray[Any]:
        if isinstance(index, (int, slice)):
            index = (index, slice(None), slice(None), slice(None))
        return self._get_tuple(index)

    @abstractmethod
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        pass


class ComplementNone(Complement):

    def __init__(self, target: Any, source: Any) -> None:
        super().__init__(target, source)

    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        index, changes = index_to_slices(index, self.shape)
        result = self._source[index]
        return apply_index_to_slices_changes(result, changes)


class ComplementNearest(Complement):

    def __init__(self, target: Any, source: Any) -> None:
        super().__init__(target, source)

        self._nearest_grid_points = nearest_grid_points(
            self._source.latitudes,
            self._source.longitudes,
            self._target.latitudes,
            self._target.longitudes,
        )

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        pass

    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        variable_index = 1
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, variable_index, slice(None))
        source_index = [self._source.name_to_index[x] for x in self.variables[previous]]
        source_data = self._source[index[0], source_index, index[2], ...]
        target_data = source_data[..., self._nearest_grid_points]

        result = target_data[..., index[3]]

        return apply_index_to_slices_changes(result, changes)


def complement_factory(args: Tuple, kwargs: dict) -> Dataset:
    from .select import Select

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
    reorder = source.variables
    complemented = _open([target, complement])
    ordered = (
        Select(complemented, complemented._reorder_to_columns(reorder), {"reoder": reorder})._subset(**kwargs).mutate(),
    )
    return ordered
