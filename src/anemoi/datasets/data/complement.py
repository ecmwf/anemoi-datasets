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

import numpy as np
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
        LOG.info(f"The following variables will be complemented: {self._variables}")

        if not self._variables:
            raise ValueError("Augment: no missing variables")

    @property
    def variables(self) -> list[str]:
        """Returns the list of variables to be added to the target dataset."""
        return self._variables

    @property
    def statistics(self) -> dict[str, NDArray[Any]]:
        datasets = [self._source, self._target]
        return {
            k: [d.statistics[k][d.name_to_index[i]] for d in datasets for i in d.variables if i in self.variables]
            for k in datasets[0].statistics
        }

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
        index = [self._source.name_to_index[v] for v in self._variables]
        if delta is None:
            delta = self.frequency
        return {k: v[index] for k, v in self._source.statistics_tendencies(delta).items()}

    @property
    def name_to_index(self) -> dict[str, int]:
        """Returns a dictionary mapping variable names to their indices."""
        return {v: i for i, v in enumerate(self.variables)}

    @property
    def shape(self) -> Shape:
        """Returns the shape of the complemented dataset."""
        shape = self._target.shape
        return (shape[0], len(self._variables)) + shape[2:]

    @property
    def variables_metadata(self) -> dict[str, Any]:
        """Returns the metadata of the variables to be added to the target dataset."""
        # Merge the two dicts first
        all_meta = {**self._source.variables_metadata, **self._target.variables_metadata}

        # Filter to keep only desired variables
        return {k: v for k, v in all_meta.items() if k in self._variables}

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
    def missing(self) -> set[int]:
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

    def __init__(self, target: Any, source: Any, max_distance: float = None, k: int = 1) -> None:
        """Initializes the ComplementNearest class.

        Parameters
        ----------
        target : Any
            The target dataset.
        source : Any
            The source dataset.
        max_distance : float, optional
            The maximum distance for nearest neighbor interpolation, default is None.
        k : int, optional
            The number of k closest neighbors to consider for interpolation
        """
        super().__init__(target, source)

        self.k = k
        self._distances, self._nearest_grid_points = nearest_grid_points(
            self._source.latitudes,
            self._source.longitudes,
            self._target.latitudes,
            self._target.longitudes,
            max_distance=max_distance,
            k=k,
        )

        if k == 1:
            self._distances = np.expand_dims(self._distances, axis=1)
            self._nearest_grid_points = np.expand_dims(self._nearest_grid_points, axis=1)

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
        if any(self._nearest_grid_points >= source_data.shape[-1]):
            target_shape = source_data.shape[:-1] + self._target.shape[-1:]
            target_data = np.full(target_shape, np.nan, dtype=self._target.dtype)
            cond = self._nearest_grid_points < source_data.shape[-1]
            reachable = np.where(cond)[0]
            nearest_reachable = self._nearest_grid_points[cond]
            target_data[..., reachable] = source_data[..., nearest_reachable]
            result = target_data[..., index[3]]
        else:
            target_data = source_data[..., self._nearest_grid_points]
            epsilon = 1e-8  # prevent division by zero
            weights = 1.0 / (self._distances + epsilon)
            weights = weights.astype(target_data.dtype)
            weights /= weights.sum(axis=1, keepdims=True)  # normalize

            # Reshape weights to broadcast correctly
            # Add leading singleton dimensions so it matches target_data shape
            while weights.ndim < target_data.ndim:
                weights = np.expand_dims(weights, axis=0)

            # Compute weighted average along the last dimension
            final_point = np.sum(target_data * weights, axis=-1)
            result = final_point[..., index[3]]

        return apply_index_to_slices_changes(result, changes)


def complement_factory(args: tuple, kwargs: dict) -> Dataset:
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

    if interpolation == "nearest":
        k = kwargs.pop("k", 1)
        max_distance = kwargs.pop("max_distance", None)
        complement = Class(target=target, source=source, k=k, max_distance=max_distance)._subset(**kwargs)

    else:
        complement = Class(target=target, source=source)._subset(**kwargs)

    joined = _open_dataset([target, complement])

    return _open_dataset(joined, reorder=sorted(joined.variables))
