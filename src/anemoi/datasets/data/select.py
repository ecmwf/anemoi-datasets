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
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from numpy.typing import NDArray

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import Node
from .debug import Source
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Select(Forwards):
    """Class to select a subset of variables from a dataset."""

    def __init__(self, dataset: Dataset, indices: List[int], reason: Dict[str, Any]) -> None:
        """Initialize the Select class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to select from.
        indices : List[int]
            The indices of the variables to select.
        reason : Dict[str, Any]
            The reason for the selection.
        """
        reason = reason.copy()

        while isinstance(dataset, Select):
            indices = [dataset.indices[i] for i in indices]
            reason.update(dataset.reason)
            dataset = dataset.dataset

        self.dataset: Dataset = dataset
        self.indices = list(indices)
        assert len(self.indices) > 0
        self.reason = reason or {"indices": self.indices}

        super().__init__(dataset)

    def clone(self, dataset: Dataset) -> Dataset:
        """Clone the Select object with a new dataset.

        Parameters
        ----------
        dataset : Dataset
            The new dataset.

        Returns
        -------
        Select
            The cloned Select object.
        """
        return self.__class__(dataset, self.indices, self.reason).mutate()

    def mutate(self) -> Dataset:
        """Mutate the dataset.

        Returns
        -------
        Dataset
            The mutated dataset.
        """
        return self.forward.swap_with_parent(parent=self)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get a tuple of data.

        Parameters
        ----------
        index : TupleIndex
            The index to retrieve.

        Returns
        -------
        NDArray[Any]
            The retrieved data.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 1, slice(None))
        result = self.dataset[index]
        result = result[:, self.indices]
        result = result[:, previous]
        result = apply_index_to_slices_changes(result, changes)
        return result

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get an item from the dataset.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve.

        Returns
        -------
        NDArray[Any]
            The retrieved data.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        row = self.dataset[n]
        if isinstance(n, slice):
            return row[:, self.indices]

        return row[self.indices]

    @cached_property
    def shape(self) -> Shape:
        """Get the shape of the dataset."""
        return (len(self), len(self.indices)) + self.dataset.shape[2:]

    @cached_property
    def variables(self) -> List[str]:
        """Get the variables of the dataset."""
        return [self.dataset.variables[i] for i in self.indices]

    @cached_property
    def variables_metadata(self) -> Dict[str, Any]:
        """Get the metadata of the variables."""
        return {k: v for k, v in self.dataset.variables_metadata.items() if k in self.variables}

    @cached_property
    def name_to_index(self) -> Dict[str, int]:
        """Get the mapping of variable names to indices."""
        return {k: i for i, k in enumerate(self.variables)}

    @cached_property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        """Get the statistics of the dataset."""
        return {k: v[self.indices] for k, v in self.dataset.statistics.items()}

    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        """Get the statistical tendencies of the dataset.

        Parameters
        ----------
        delta : Optional[datetime.timedelta]
            The time delta for the tendencies.

        Returns
        -------
        Dict[str, NDArray[Any]]
            The statistical tendencies.
        """
        if delta is None:
            delta = self.frequency
        return {k: v[self.indices] for k, v in self.dataset.statistics_tendencies(delta).items()}

    def metadata_specific(self, **kwargs: Any) -> Dict[str, Any]:
        """Get the specific metadata of the dataset.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Dict[str, Any]
            The specific metadata.
        """
        return super().metadata_specific(indices=self.indices, **kwargs)

    def source(self, index: int) -> Source:
        """Get the source of the dataset.

        Parameters
        ----------
        index : int
            The index of the source.

        Returns
        -------
        Source
            The source of the dataset.
        """
        return Source(self, index, self.dataset.source(self.indices[index]))

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.dataset.tree()], **self.reason)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the subclass.
        """
        # return dict(indices=self.indices)
        return dict(reason=self.reason)


class Rename(Forwards):
    """Class to rename variables in a dataset."""

    def __init__(self, dataset: Dataset, rename: Dict[str, str]) -> None:
        """Initialize the Rename class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to rename.
        rename : Dict[str, str]
            The mapping of old names to new names.
        """
        super().__init__(dataset)
        for n in rename:
            assert n in dataset.variables, n

        self._variables = [rename.get(v, v) for v in dataset.variables]
        self._variables_metadata = {rename.get(k, k): v for k, v in dataset.variables_metadata.items()}

        self.rename = rename

    @property
    def variables(self) -> List[str]:
        """Get the renamed variables."""
        return self._variables

    @property
    def variables_metadata(self) -> Dict[str, Any]:
        """Get the renamed variables metadata."""
        return self._variables_metadata

    @cached_property
    def name_to_index(self) -> Dict[str, int]:
        """Get the mapping of renamed variable names to indices."""
        return {k: i for i, k in enumerate(self.variables)}

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns:
            Node: The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], rename=self.rename)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the subclass.

        Returns:
            Dict[str, Any]: The metadata specific to the subclass.
        """
        return dict(rename=self.rename)
