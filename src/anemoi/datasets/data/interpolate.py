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
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class InterpolateFrequency(Forwards):
    """A class to represent a dataset with interpolated frequency."""

    def __init__(self, dataset: Dataset, frequency: str) -> None:
        """Initialize the InterpolateFrequency class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be interpolated.
        frequency : str
            The interpolation frequency.
        """
        super().__init__(dataset)
        self._frequency = frequency_to_timedelta(frequency)

        self.seconds = self._frequency.total_seconds()
        other_seconds = dataset.frequency.total_seconds()

        self.seconds = int(self.seconds)
        assert self.seconds == self._frequency.total_seconds()

        other_seconds = int(other_seconds)
        assert other_seconds == dataset.frequency.total_seconds()

        if self.seconds >= other_seconds:
            raise ValueError(
                f"Interpolate frequency {self._frequency} must be more frequent than dataset frequency {dataset.frequency}"
            )

        if other_seconds % self.seconds != 0:
            raise ValueError(
                f"Interpolate frequency {self._frequency}  must be a multiple of the dataset frequency {dataset.frequency}"
            )

        self.ratio = other_seconds // self.seconds
        self.alphas = np.linspace(0, 1, self.ratio + 1)
        self.other_len = len(dataset)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get the interpolated data for a tuple index.

        Parameters
        ----------
        index : TupleIndex
            The tuple index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The interpolated data for the tuple index.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 0, slice(None))
        result = self._get_slice(previous)
        return apply_index_to_slices_changes(result[index], changes)

    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Get the interpolated data for a slice.

        Parameters
        ----------
        s : slice
            The slice to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The interpolated data for the slice.
        """
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get the interpolated data at the specified index.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The interpolated data at the specified index.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        if n < 0:
            n += self._len

        if n == self._len - 1:
            # Special case for the last element
            return self.forward[-1]

        i = n // self.ratio
        x = n % self.ratio

        if x == 0:
            # No interpolation needed
            return self.forward[i]

        alpha = self.alphas[x]

        assert 0 < alpha < 1, alpha
        return self.forward[i] * (1 - alpha) + self.forward[i + 1] * alpha

    def __len__(self) -> int:
        """Get the length of the interpolated dataset.

        Returns
        -------
        int
            The length of the interpolated dataset.
        """
        return (self.other_len - 1) * self.ratio + 1

    @property
    def frequency(self) -> datetime.timedelta:
        """Get the interpolation frequency."""
        return self._frequency

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Get the interpolated dates."""
        result = []
        deltas = [np.timedelta64(self.seconds * i, "s") for i in range(self.ratio)]
        for d in self.forward.dates[:-1]:
            for i in deltas:
                result.append(d + i)
        result.append(self.forward.dates[-1])
        return np.array(result)

    @property
    def shape(self) -> Shape:
        """Get the shape of the interpolated dataset."""
        return (self._len,) + self.forward.shape[1:]

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], frequency=self.frequency)

    @cached_property
    def missing(self) -> Set[int]:
        """Get the missing data indices."""
        result = []
        j = 0
        for i in range(self.other_len):
            missing = i in self.forward.missing
            for _ in range(self.ratio):
                if missing:
                    result.append(j)
                j += 1

        result = set(x for x in result if x < self._len)
        return result

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the InterpolateFrequency subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the InterpolateFrequency subclass.
        """
        return {
            # "frequency": frequency_to_string(self._frequency),
        }


class InterpolateNearest(Forwards):
    def __init__(
        self, dataset: Dataset, interpolate_variables: List[str], max_distance: Optional[float] = None
    ) -> None:
        """Initialize the InterpolateNearest class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be interpolated.
        interpolate_variables : List[str]
            The variables to be interpolated.
        max_distance : Optional[float], optional
            The maximum distance for nearest neighbor search, by default None.
        """
        from ..grids import nearest_grid_points

        super().__init__(dataset)
        self.vars = interpolate_variables
        self.var_mask = {
            v: np.where(~np.isnan(dataset[0, dataset.name_to_index[v]].squeeze()))[0] for v in interpolate_variables
        }
        self.dataset = dataset
        self.ngp_var = {
            v: nearest_grid_points(
                dataset.latitudes[self.var_mask[v]],
                dataset.longitudes[self.var_mask[v]],
                dataset.latitudes,
                dataset.longitudes,
                max_distance=max_distance,
            )
            for v in interpolate_variables
        }

    def tree(self) -> Node:
        return Node(self, [self.forward.tree()], interpolate_variables=self.vars)

    @property
    def shape(self) -> Shape:
        return self.forward.shape

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.forward.metadata()

    @staticmethod
    def slice_len(slice_obj: slice) -> int:
        # Compute the length of the slice
        return max(0, (slice_obj.stop - slice_obj.start + (slice_obj.step or 1) - 1) // (slice_obj.step or 1))

    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        index, changes = index_to_slices(index, self.shape)
        source_data = self.forward[index[0]]
        target_data = source_data.copy()
        vars_to_interpolate = [self.forward.name_to_index[v] for v in self.vars]
        for v, i in zip(self.vars, vars_to_interpolate):
            target_data[:, i, ...] = source_data[:, i][..., self.var_mask[v][self.ngp_var[v]]]
        result = target_data[(slice(None),) + index[1:]]
        return apply_index_to_slices_changes(result, changes)

    def __getitem__(self, index: Union[int, slice, Tuple[Union[int, slice], ...]]) -> NDArray[Any]:
        if isinstance(index, (int, slice)):
            index = (index, slice(None), slice(None), slice(None))
        return self._get_tuple(index)

    def subclass_metadata_specific(self) -> Dict[str, Any]:
        return {
            "interpolate_variables": self.vars,
        }
