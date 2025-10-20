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
from collections.abc import Sequence
from functools import cached_property
from typing import Any
from typing import Union

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.data.dataset import Dataset
from anemoi.datasets.data.dataset import FullIndex
from anemoi.datasets.data.dataset import Shape
from anemoi.datasets.data.dataset import TupleIndex
from anemoi.datasets.data.debug import Node
from anemoi.datasets.data.debug import Source
from anemoi.datasets.data.debug import debug_indexing
from anemoi.datasets.data.forwards import Forwards
from anemoi.datasets.data.indexing import apply_index_to_slices_changes
from anemoi.datasets.data.indexing import expand_list_indexing
from anemoi.datasets.data.indexing import index_to_slices
from anemoi.datasets.data.indexing import make_slice_or_index_from_list_or_tuple
from anemoi.datasets.data.indexing import update_tuple

LOG = logging.getLogger(__name__)


def _default(a: int, b: int, dates: NDArray[np.datetime64]) -> list[int]:
    """Default combination function for reasons.

    Parameters:
    a (int): First integer value.
    b (int): Second integer value.
    dates (NDArray[np.datetime64]): Array of datetime64 dates.

    Returns:
    list[int]: List containing the two input integers.
    """
    return [a, b]


def _start(a: int, b: int, dates: NDArray[np.datetime64]) -> int:
    """Determine the start date between two dates.

    Parameters:
    a (int): First integer value.
    b (int): Second integer value.
    dates (NDArray[np.datetime64]): Array of datetime64 dates.

    Returns:
    int: The index of the start date.
    """
    from anemoi.datasets.data.misc import as_first_date

    c = as_first_date(a, dates)
    d = as_first_date(b, dates)
    if c < d:
        return b
    else:
        return a


def _end(a: int, b: int, dates: NDArray[np.datetime64]) -> int:
    """Determine the end date between two dates.

    Parameters:
    a (int): First integer value.
    b (int): Second integer value.
    dates (NDArray[np.datetime64]): Array of datetime64 dates.

    Returns:
    int: The index of the end date.
    """
    from anemoi.datasets.data.misc import as_last_date

    c = as_last_date(a, dates)
    d = as_last_date(b, dates)
    if c < d:
        return a
    else:
        return b


def _combine_reasons(reason1: dict[str, Any], reason2: dict[str, Any], dates: NDArray[np.datetime64]) -> dict[str, Any]:
    """Combine two reason dictionaries.

    Parameters:
    reason1 (Dict[str, Any]): First reason dictionary.
    reason2 (Dict[str, Any]): Second reason dictionary.
    dates (NDArray[np.datetime64]): Array of datetime64 dates.

    Returns:
    Dict[str, Any]: Combined reason dictionary.
    """

    reason = reason1.copy()
    for k, v in reason2.items():
        if k not in reason:
            reason[k] = v
        else:
            func = globals().get(f"_{k}", _default)
            reason[k] = func(reason[k], v, dates)
    return reason


class Subset(Forwards):
    """Select a subset of the dates.

    Attributes:
    dataset (Dataset): The dataset.
    indices (List[int]): List of indices.
    reason (Dict[str, Any]): Dictionary of reasons.
    """

    def __init__(self, dataset: Union[Dataset, "Subset"], indices: Sequence[int], reason: dict[str, Any]) -> None:
        """Initialize the Subset.

        Parameters:
        dataset (Dataset | Subset): The dataset or subset.
        indices (Sequence[int]): Sequence of indices.
        reason (Dict[str, Any]): Dictionary of reasons.
        """
        while isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices]
            reason = _combine_reasons(reason, dataset.reason, dataset.dates)
            dataset = dataset.dataset

        self.dataset: Dataset = dataset
        self.indices: list[int] = list(indices)
        self.reason: dict[str, Any] = {k: v for k, v in reason.items() if v is not None}

        # Forward other properties to the super dataset
        super().__init__(dataset)

    def clone(self, dataset: Dataset) -> Dataset:
        """Clone the subset with a new dataset.

        Parameters:
        dataset (Dataset): The new dataset.

        Returns:
        Dataset: The cloned subset.
        """
        return self.__class__(dataset, self.indices, self.reason).mutate()

    def mutate(self) -> Dataset:
        """Mutate the subset.

        Returns:
        Dataset: The mutated subset.
        """
        return self.forward.swap_with_parent(parent=self)

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get item by index.

        Parameters:
        n (FullIndex): The index.

        Returns:
        NDArray[Any]: The indexed data.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        assert n >= 0, n
        n = self.indices[n]
        return self.dataset[n]

    def get_aux(self, n: FullIndex) -> NDArray[Any]:
        assert n >= 0, n
        n = self.indices[n]
        return self.dataset.get_aux(n)

    @debug_indexing
    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Get slice of data.

        Parameters:
        s (slice): The slice.

        Returns:
        NDArray[Any]: The sliced data.
        """
        # TODO: check if the indices can be simplified to a slice
        # the time checking maybe be longer than the time saved
        # using a slice
        indices = [self.indices[i] for i in range(*s.indices(self._len))]
        indices = make_slice_or_index_from_list_or_tuple(indices)
        if isinstance(indices, slice):
            return self.dataset[indices]
        return np.stack([self.dataset[i] for i in indices])

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, n: TupleIndex) -> NDArray[Any]:
        """Get tuple of data.

        Parameters:
        n (TupleIndex): The tuple index.

        Returns:
        NDArray[Any]: The indexed data.
        """
        index, changes = index_to_slices(n, self.shape)
        indices = [self.indices[i] for i in range(*index[0].indices(self._len))]
        indices = make_slice_or_index_from_list_or_tuple(indices)
        index, _ = update_tuple(index, 0, indices)
        result = self.dataset[index]
        result = apply_index_to_slices_changes(result, changes)
        return result

    def __len__(self) -> int:
        """Get the length of the subset.

        Returns:
        int: The length of the subset.
        """
        return len(self.indices)

    @cached_property
    def shape(self) -> Shape:
        """Get the shape of the subset."""
        return (len(self),) + self.dataset.shape[1:]

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Get the dates of the subset."""
        return self.dataset.dates[self.indices]

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Get the frequency of the subset."""
        dates = self.dates
        if len(dates) < 2:
            raise ValueError(f"Cannot determine frequency of a subset with less than two dates ({self.dates}).")
        return frequency_to_timedelta(dates[1].astype(object) - dates[0].astype(object))

    def source(self, index: int) -> Source:
        """Get the source of the subset.

        Parameters:
        index (int): The index.

        Returns:
        Source: The source of the subset.
        """
        return Source(self, index, self.forward.source(index))

    def __repr__(self) -> str:
        """Get the string representation of the subset.

        Returns:
        str: The string representation of the subset.
        """
        return f"Subset({self.dataset},{self.dates[0]}...{self.dates[-1]}/{self.frequency})"

    @cached_property
    def missing(self) -> set[int]:
        """Get the missing indices of the subset."""
        missing = self.dataset.missing
        result: set[int] = set()
        for j, i in enumerate(self.indices):
            if i in missing:
                result.add(j)
        return result

    def tree(self) -> Node:
        """Get the tree representation of the subset.

        Returns:
        Node: The tree representation of the subset.
        """
        return Node(self, [self.dataset.tree()], **self.reason)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns:
        Dict[str, Any]: The metadata specific to the forwards subclass.
        """
        return {
            # "indices": self.indices,
            "reason": self.reason,
        }

    def forward_subclass_origin(self, index):
        assert (
            isinstance(index, tuple) and len(index) == 4 and all(a > b >= 0 for a, b in zip(self.shape, index))
        ), tuple
        return self.dataset.origin((self.indices[index[0]], index[1], index[2], index[3]))

    def project(self, projection):
        projection = projection.from_indices(axis=0, indices=self.indices)
        return self.dataset.project(projection)
