# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from abc import abstractmethod
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
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
from .misc import _open_dataset

LOG = logging.getLogger(__name__)


class Complement(Combined):
    """A class to complement a target dataset with variables from a source dataset,
    interpolated on the grid of the target dataset.

    Attributes
    ----------
    target : Dataset
        The target dataset.
    source : Dataset
        The source dataset.
    variables : List[str]
        List of variables to be added to the target dataset.
    """

    def __init__(
        self,
        target: Dataset,
        source: Dataset,
        what: str = "variables",
        interpolation: str = "nearest",
    ) -> None:
        """Initializes the Complement class.

        Parameters
        ----------
        target : Dataset
            The target dataset.
        source : Dataset
            The source dataset.
        what : str, optional
            What to complement, default is "variables".
        interpolation : str, optional
            Interpolation method, default is "nearest".
        """
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
        """Returns the list of variables to be added to the target dataset."""
        return self._variables

    @property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        """Returns the statistics of the complemented dataset."""
        index = [self._source.name_to_index[v] for v in self._variables]
        return {k: v[index] for k, v in self._source.statistics.items()}

    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        index = [self._source.name_to_index[v] for v in self._variables]
        if delta is None:
            delta = self.frequency
        return {k: v[index] for k, v in self._source.statistics_tendencies(delta).items()}

    @property
    def name_to_index(self) -> Dict[str, int]:
        """Returns a dictionary mapping variable names to their indices."""
        return {v: i for i, v in enumerate(self.variables)}

    @property
    def shape(self) -> Shape:
        """Returns the shape of the complemented dataset."""
        shape = self._target.shape
        return (shape[0], len(self._variables)) + shape[2:]

    @property
    def variables_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the variables to be added to the target dataset."""
        return {k: v for k, v in self._source.variables_metadata.items() if k in self._variables}

    def check_same_variables(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the variables in two datasets are the same.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        pass

    @cached_property
    def missing(self) -> Set[int]:
        """Returns the set of missing indices in the source and target datasets."""
        missing = self._source.missing.copy()
        missing = missing | self._target.missing
        return set(missing)

    def tree(self) -> Node:
        """Generates a hierarchical tree structure for the Complement instance and its associated datasets.

        Returns
        -------
        Node
            A Node object representing the Complement instance as the root node, with each dataset in self.datasets represented as a child node.
        """
        return Node(self, [d.tree() for d in (self._target, self._source)])

    def __getitem__(self, index: FullIndex) -> NDArray[Any]:
        """Gets the data at the specified index.

        Parameters
        ----------
        index : FullIndex
            The index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The data at the specified index.
        """
        if isinstance(index, (int, slice)):
            index = (index, slice(None), slice(None), slice(None))
        return self._get_tuple(index)

    @abstractmethod
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Abstract method to get the data at the specified tuple index.

        Parameters
        ----------
        index : TupleIndex
            The tuple index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The data at the specified tuple index.
        """
        pass

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns
        -------
        dict[str, Any]
            The metadata specific to the forwards subclass.
        """
        return dict(complement=self._source.dataset_metadata())


class ComplementNone(Complement):
    """A class to complement a target dataset with variables from a source dataset without interpolation."""

    def __init__(self, target: Any, source: Any) -> None:
        """Initializes the ComplementNone class.

        Parameters
        ----------
        target : Any
            The target dataset.
        source : Any
            The source dataset.
        """
        super().__init__(target, source)

    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Gets the data at the specified tuple index without interpolation.

        Parameters
        ----------
        index : TupleIndex
            The tuple index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The data at the specified tuple index.
        """
        index, changes = index_to_slices(index, self.shape)
        result = self._source[index]
        return apply_index_to_slices_changes(result, changes)


class ComplementNearest(Complement):
    """A class to complement a target dataset with variables from a source dataset using nearest neighbor interpolation."""

    def __init__(self, target: Any, source: Any, max_distance: float = None) -> None:
        """Initializes the ComplementNearest class.

        Parameters
        ----------
        target : Any
            The target dataset.
        source : Any
            The source dataset.
        max_distance : float, optional
            The maximum distance for nearest neighbor interpolation, default is None.
        """
        super().__init__(target, source)

        self._nearest_grid_points = nearest_grid_points(
            self._source.latitudes,
            self._source.longitudes,
            self._target.latitudes,
            self._target.longitudes,
            max_distance=max_distance,
        )

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Checks the compatibility of two datasets for nearest neighbor interpolation.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        pass

    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Gets the data at the specified tuple index using nearest neighbor interpolation.

        Parameters
        ----------
        index : TupleIndex
            The tuple index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The data at the specified tuple index.
        """
        variable_index = 1
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, variable_index, slice(None))
        source_index = [self._source.name_to_index[x] for x in self.variables[previous]]
        source_data = self._source[index[0], source_index, index[2], ...]
        target_data = source_data[..., self._nearest_grid_points]

        result = target_data[..., index[3]]

        return apply_index_to_slices_changes(result, changes)


def complement_factory(args: Tuple, kwargs: dict) -> Dataset:
    """Factory function to create a Complement instance based on the provided arguments.

    Parameters
    ----------
    args : Tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    Dataset
        The complemented dataset.
    """

    assert len(args) == 0, args

    source = kwargs.pop("source")
    target = kwargs.pop("complement")
    what = kwargs.pop("what", "variables")
    interpolation = kwargs.pop("interpolation", "none")

    if what != "variables":
        raise NotImplementedError(f"Complement what={what} not implemented")

    if interpolation not in ("none", "nearest"):
        raise NotImplementedError(f"Complement method={interpolation} not implemented")

    source = _open_dataset(source)
    target = _open_dataset(target)
    # `select` is the same as `variables`
    (source, target), kwargs = _auto_adjust([source, target], kwargs, exclude=["select"])

    Class = {
        None: ComplementNone,
        "none": ComplementNone,
        "nearest": ComplementNearest,
    }[interpolation]

    complement = Class(target=target, source=source)._subset(**kwargs)

    return _open_dataset([target, complement], reorder=source.variables)
