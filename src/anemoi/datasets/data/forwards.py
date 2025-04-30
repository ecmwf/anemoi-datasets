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
import warnings
from abc import abstractmethod
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy as np
from numpy.typing import NDArray

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import debug_indexing
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import length_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Forwards(Dataset):
    """A class to represent a dataset that forwards its properties and methods to another dataset."""

    def __init__(self, forward: Dataset) -> None:
        """Initializes a Forwards object.

        Parameters
        ----------
        forward : Dataset
            The forward dataset.
        """
        self.forward = forward.mutate()

    def __len__(self) -> int:
        """Returns the length of the forward dataset.

        Returns
        -------
        int
            Length of the forward dataset.
        """
        return len(self.forward)

    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Retrieves data from the forward dataset based on the given index.

        Parameters
        ----------
        n : Index
            Index specifying the data to retrieve.

        Returns
        -------
        Any
            Data from the forward dataset based on the index.
        """
        return self.forward[n]

    @property
    def name(self) -> Optional[str]:
        """Returns the name of the forward dataset."""
        if self._name is not None:
            return self._name
        return self.forward.name

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Returns the dates of the forward dataset."""
        return self.forward.dates

    @property
    def resolution(self) -> str:
        """Returns the resolution of the forward dataset."""
        return self.forward.resolution

    @property
    def field_shape(self) -> Shape:
        """Returns the field shape of the forward dataset."""
        return self.forward.field_shape

    @property
    def frequency(self) -> datetime.timedelta:
        """Returns the frequency of the forward dataset."""
        return self.forward.frequency

    @property
    def latitudes(self) -> NDArray[Any]:
        """Returns the latitudes of the forward dataset."""
        return self.forward.latitudes

    @property
    def longitudes(self) -> NDArray[Any]:
        """Returns the longitudes of the forward dataset."""
        return self.forward.longitudes

    @property
    def name_to_index(self) -> Dict[str, int]:
        """Returns a dictionary mapping variable names to their indices."""
        return self.forward.name_to_index

    @property
    def variables(self) -> List[str]:
        """Returns the variables of the forward dataset."""
        return self.forward.variables

    @property
    def variables_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the variables in the forward dataset."""
        return self.forward.variables_metadata

    @property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        """Returns the statistics of the forward dataset."""
        return self.forward.statistics

    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        """Returns the statistics tendencies of the forward dataset.

        Parameters
        ----------
        delta : Optional[Any]
            Time delta for calculating tendencies.

        Returns
        -------
        Any
            Statistics tendencies of the forward dataset.
        """
        if delta is None:
            delta = self.frequency
        return self.forward.statistics_tendencies(delta)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the forward dataset."""
        return self.forward.shape

    @property
    def dtype(self) -> Any:
        """Returns the data type of the forward dataset."""
        return self.forward.dtype

    @property
    def missing(self) -> Set[int]:
        """Returns the missing data information of the forward dataset."""
        return self.forward.missing

    @property
    def grids(self) -> Any:
        """Returns the grids of the forward dataset."""
        return self.forward.grids

    def metadata_specific(self, **kwargs: Any) -> Dict[str, Any]:
        """Returns metadata specific to the forward dataset.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the forward dataset.
        """
        return super().metadata_specific(
            forward=self.forward.metadata_specific(),
            **self.forwards_subclass_metadata_specific(),
            **kwargs,
        )

    def collect_supporting_arrays(self, collected: List[Any], *path: Any) -> None:
        """Collects supporting arrays from the forward dataset.

        Parameters
        ----------
        collected : List[Any]
            List to which the supporting arrays are appended.
        *path : Any
            Variable length argument list specifying the paths for the arrays.
        """
        self.forward.collect_supporting_arrays(collected, *path)

    def collect_input_sources(self, collected: List[Any]) -> None:
        """Collects input sources from the forward dataset.

        Parameters
        ----------
        collected : List[Any]
            List to which the input sources are appended.
        """
        self.forward.collect_input_sources(collected)

    def source(self, index: int) -> Any:
        """Returns the source of the data at the specified index.

        Parameters
        ----------
        index : int
            Index specifying the data source.

        Returns
        -------
        Any
            Source of the data at the specified index.
        """
        return self.forward.source(index)

    @abstractmethod
    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Returns metadata specific to the subclass."""
        pass

    def get_dataset_names(self, names: Set[str]) -> None:
        """Collects the names of the datasets.

        Parameters
        ----------
        names : set
            Set to which the dataset names are added.
        """
        self.forward.get_dataset_names(names)

    @property
    def constant_fields(self) -> List[str]:
        """Returns the constant fields of the forward dataset."""
        return self.forward.constant_fields


class Combined(Forwards):
    """A class to combine multiple datasets into a single dataset."""

    def __init__(self, datasets: List[Dataset]) -> None:
        """Initializes a Combined object.

        Parameters
        ----------
        datasets : List[Dataset]
            List of datasets to be combined.
        """
        self.datasets = datasets
        assert len(self.datasets) > 1, len(self.datasets)

        for d in self.datasets[1:]:
            self.check_compatibility(self.datasets[0], d)

        # Forward most properties to the first dataset
        super().__init__(datasets[0])

    def mutate(self) -> Dataset:
        """Returns the mutated dataset.

        Returns
        -------
        Dataset
            Mutated dataset.
        """
        return self

    def check_same_resolution(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the resolutions of two datasets are the same.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the resolutions are not the same.
        """
        if d1.resolution is None or d2.resolution is None:
            LOG.warning("One of the datasets has no resolution, cannot check compatibility")
            return

        if d1.resolution != d2.resolution:
            raise ValueError(f"Incompatible resolutions: {d1.resolution} and {d2.resolution} ({d1} {d2})")

    def check_same_frequency(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the frequencies of two datasets are the same.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the frequencies are not the same.
        """
        if d1.frequency != d2.frequency:
            raise ValueError(f"Incompatible frequencies: {d1.frequency} and {d2.frequency} ({d1} {d2})")

    def check_same_grid(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the grids of two datasets are the same.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the grids are not the same.
        """
        if (d1.latitudes != d2.latitudes).any() or (d1.longitudes != d2.longitudes).any():
            raise ValueError(f"Incompatible grid ({d1} {d2})")

    def check_same_shape(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the shapes of two datasets are the same.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the shapes are not the same.
        """
        if d1.shape[1:] != d2.shape[1:]:
            raise ValueError(f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})")

        if d1.variables != d2.variables:
            raise ValueError(f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})")

    def check_same_sub_shapes(self, d1: Any, d2: Any, drop_axis: int) -> None:
        """Checks if the sub-shapes of two datasets are the same along a given axis.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.
        drop_axis : int
            Axis along which to check the sub-shapes.

        Raises
        ------
        ValueError
            If the sub-shapes are not the same.
        """
        shape1 = d1.sub_shape(drop_axis)
        shape2 = d2.sub_shape(drop_axis)

        if shape1 != shape2:
            raise ValueError(f"Incompatible shapes: {d1.shape} and {d2.shape} ({d1} {d2})")

    def check_same_variables(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the variables of two datasets are the same.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the variables are not the same.
        """
        if d1.variables != d2.variables:
            raise ValueError(f"Incompatible variables: {d1.variables} and {d2.variables} ({d1} {d2})")

    def check_same_lengths(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the lengths of two datasets are the same.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the lengths are not the same.
        """
        if d1._len != d2._len:
            raise ValueError(f"Incompatible lengths: {d1._len} and {d2._len}")

    def check_same_dates(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the dates of two datasets are the same.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the dates are not the same.
        """
        self.check_same_frequency(d1, d2)

        if d1.dates[0] != d2.dates[0]:
            raise ValueError(f"Incompatible start dates: {d1.dates[0]} and {d2.dates[0]} ({d1} {d2})")

        if d1.dates[-1] != d2.dates[-1]:
            raise ValueError(f"Incompatible end dates: {d1.dates[-1]} and {d2.dates[-1]} ({d1} {d2})")

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if two datasets are compatible.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the datasets are not compatible.
        """
        # These are the default checks
        # Derived classes should turn individual checks off if they are not needed
        self.check_same_resolution(d1, d2)
        self.check_same_frequency(d1, d2)
        self.check_same_grid(d1, d2)
        self.check_same_lengths(d1, d2)
        self.check_same_variables(d1, d2)
        self.check_same_dates(d1, d2)

    def provenance(self) -> List[Any]:
        """Returns the provenance of the combined datasets.

        Returns
        -------
        List[Any]
            Provenance of the combined datasets.
        """
        return [d.provenance() for d in self.datasets]

    def __repr__(self) -> str:
        """Returns a string representation of the Combined object.

        Returns
        -------
        str
            String representation of the Combined object.
        """
        lst = ", ".join(repr(d) for d in self.datasets)
        return f"{self.__class__.__name__}({lst})"

    def metadata_specific(self, **kwargs: Any) -> Dict[str, Any]:
        """Returns metadata specific to the combined datasets.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            Metadata specific to the combined datasets.
        """
        # We need to skip the forward superclass
        # TODO: revisit this
        return Dataset.metadata_specific(
            self,
            datasets=[d.metadata_specific() for d in self.datasets],
            **kwargs,
        )

    def collect_supporting_arrays(self, collected: List[Any], *path: Any) -> None:
        """Collects supporting arrays from the combined datasets.

        Parameters
        ----------
        collected : List[Any]
            List to which the supporting arrays are appended.
        *path : Any
            Variable length argument list specifying the paths for the arrays.
        """
        warnings.warn(f"The behaviour of {self.__class__.__name__}.collect_supporting_arrays() is not well defined")
        for i, d in enumerate(self.datasets):
            name = d.name if d.name is not None else i
            d.collect_supporting_arrays(collected, *path, name)

    @property
    def missing(self) -> Set[int]:
        """Returns the missing data information of the combined datasets.

        Raises
        ------
        NotImplementedError
            If the method is not implemented for Combined.
        """
        raise NotImplementedError("missing() not implemented for Combined")

    def get_dataset_names(self, names: Set[str]) -> None:
        """Collects the names of the combined datasets.

        Parameters
        ----------
        names : set
            Set to which the dataset names are added.
        """
        for d in self.datasets:
            d.get_dataset_names(names)


class GivenAxis(Combined):
    """A class to combine datasets along a given axis."""

    def __init__(self, datasets: List[Any], axis: int) -> None:
        """Initializes a GivenAxis object.

        Parameters
        ----------
        datasets : List[Any]
            List of datasets to be combined.
        axis : int
            Axis along which to combine the datasets.
        """
        self.axis = axis
        super().__init__(datasets)

        assert axis > 0 and axis < len(self.datasets[0].shape), (
            axis,
            self.datasets[0].shape,
        )

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if two datasets are compatible along the given axis.

        Parameters
        ----------
        d1 : Any
            First dataset.
        d2 : Any
            Second dataset.

        Raises
        ------
        ValueError
            If the datasets are not compatible along the given axis.
        """
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=self.axis)

    @cached_property
    def shape(self) -> Shape:
        """Returns the shape of the combined dataset along the given axis."""
        shapes = [d.shape for d in self.datasets]
        before = shapes[0][: self.axis]
        after = shapes[0][self.axis + 1 :]
        result = before + (sum(s[self.axis] for s in shapes),) + after
        assert False not in result, result
        return result

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Retrieves data from the combined dataset based on the given index.

        Parameters
        ----------
        index : Union[int, slice, Tuple[Union[int, slice], ...]]
            Index specifying the data to retrieve.

        Returns
        -------
        NDArray[Any]
            Data from the combined dataset based on the index.
        """
        index, changes = index_to_slices(index, self.shape)
        lengths = [d.shape[self.axis] for d in self.datasets]
        slices = length_to_slices(index[self.axis], lengths)
        result = [d[update_tuple(index, self.axis, i)[0]] for (d, i) in zip(self.datasets, slices) if i is not None]
        result = np.concatenate(result, axis=self.axis)
        return apply_index_to_slices_changes(result, changes)

    @debug_indexing
    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Retrieves a slice of data from the combined dataset.

        Parameters
        ----------
        s : slice
            Slice specifying the data to retrieve.

        Returns
        -------
        NDArray[Any]
            Slice of data from the combined dataset.
        """
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Retrieves data from the combined dataset based on the given index.

        Parameters
        ----------
        n : Union[int, slice, Tuple[Union[int, slice], ...]]
            Index specifying the data to retrieve.

        Returns
        -------
        NDArray[Any]
            Data from the combined dataset based on the index.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        return np.concatenate([d[n] for d in self.datasets], axis=self.axis - 1)

    @cached_property
    def missing(self) -> Set[int]:
        """Returns the missing data information of the combined dataset along the given axis."""
        offset = 0
        result: Set[int] = set()
        for d in self.datasets:
            result.update(offset + m for m in d.missing)
            if self.axis == 0:  # Advance if axis is time
                offset += len(d)
        return result
