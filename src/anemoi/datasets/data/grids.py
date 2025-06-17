# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import Node
from .debug import debug_indexing
from .forwards import Combined
from .forwards import GivenAxis
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import length_to_slices
from .indexing import update_tuple
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Concat(Combined):
    """A class to represent concatenated datasets."""

    def __len__(self) -> int:
        """Returns the total length of the concatenated datasets.

        Returns
        -------
        int
            Total length of the concatenated datasets.
        """
        return sum(len(i) for i in self.datasets)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Retrieves a tuple of data from the concatenated datasets based on the given index.

        Parameters
        ----------
        index : TupleIndex
            Index specifying the data to retrieve.

        Returns
        -------
        NDArray[Any]
            Concatenated data array from the specified index.
        """
        index, changes = index_to_slices(index, self.shape)
        # print(index, changes)
        lengths = [d.shape[0] for d in self.datasets]
        slices = length_to_slices(index[0], lengths)
        # print("slies", slices)
        result = [d[update_tuple(index, 0, i)[0]] for (d, i) in zip(self.datasets, slices) if i is not None]
        result = np.concatenate(result, axis=0)
        return apply_index_to_slices_changes(result, changes)

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Retrieves data from the concatenated datasets based on the given index.

        Parameters
        ----------
        n : FullIndex
            Index specifying the data to retrieve.

        Returns
        -------
        NDArray[Any]
            Data array from the concatenated datasets based on the index.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        # TODO: optimize
        k = 0
        while n >= self.datasets[k]._len:
            n -= self.datasets[k]._len
            k += 1
        return self.datasets[k][n]

    @debug_indexing
    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Retrieves a slice of data from the concatenated datasets.

        Parameters
        ----------
        s : slice
            Slice object specifying the range of data to retrieve.

        Returns
        -------
        NDArray[Any]
            Concatenated data array from the specified slice.
        """
        result = []

        lengths = [d.shape[0] for d in self.datasets]
        slices = length_to_slices(s, lengths)

        result = [d[i] for (d, i) in zip(self.datasets, slices) if i is not None]

        return np.concatenate(result)

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Check the compatibility of two datasets for concatenation.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=0)

    def check_same_lengths(self, d1: Dataset, d2: Dataset) -> None:
        """Check if the lengths of two datasets are the same.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        # Turned off because we are concatenating along the first axis
        pass

    def check_same_dates(self, d1: Dataset, d2: Dataset) -> None:
        """Check if the dates of two datasets are the same.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        # Turned off because we are concatenating along the dates axis
        pass

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Returns the concatenated dates of all datasets."""
        return np.concatenate([d.dates for d in self.datasets])

    @property
    def shape(self) -> Shape:
        """Returns the shape of the concatenated datasets."""
        return (len(self),) + self.datasets[0].shape[1:]

    def tree(self) -> Node:
        """Generates a hierarchical tree structure for the concatenated datasets.

        Returns
        -------
        Node
            A Node object representing the concatenated datasets.
        """
        return Node(self, [d.tree() for d in self.datasets])


class GridsBase(GivenAxis):
    """A base class for handling grids in datasets."""

    def __init__(self, datasets: List[Any], axis: int) -> None:
        """Initializes a GridsBase object.

        Parameters
        ----------
        datasets : List[Any]
            List of datasets.
        axis : int
            Axis along which to combine the datasets.
        """
        super().__init__(datasets, axis)
        # Shape: (dates, variables, ensemble, 1d-values)
        assert len(datasets[0].shape) == 4, "Grids must be 1D for now"

    def check_same_grid(self, d1: Dataset, d2: Dataset) -> None:
        """Check if the grids of two datasets are the same.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        # We don't check the grid, because we want to be able to combine
        pass

    def check_same_resolution(self, d1: Dataset, d2: Dataset) -> None:
        """Check if the resolutions of two datasets are the same.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        # We don't check the resolution, because we want to be able to combine
        pass

    def metadata_specific(self, **kwargs: Any) -> Dict[str, Any]:
        """Returns metadata specific to the GridsBase object.

        Parameters
        ----------
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the GridsBase object.
        """
        return super().metadata_specific(
            multi_grids=True,
        )

    def collect_input_sources(self, collected: List[Any]) -> None:
        """Collects input sources from the datasets.

        Parameters
        ----------
        collected : List[Any]
            List to which the input sources are appended.
        """
        # We assume that,because they have different grids, they have different input sources
        for d in self.datasets:
            collected.append(d)
            d.collect_input_sources(collected)


class Grids(GridsBase):
    """A class to represent combined grids from multiple datasets."""

    # TODO: select the statistics of the most global grid?
    @property
    def latitudes(self) -> NDArray[Any]:
        """Returns the concatenated latitudes of all datasets."""
        return np.concatenate([d.latitudes for d in self.datasets])

    @property
    def longitudes(self) -> NDArray[Any]:
        """Returns the concatenated longitudes of all datasets."""
        return np.concatenate([d.longitudes for d in self.datasets])

    @property
    def grids(self) -> Tuple[Any, ...]:
        """Returns the grids of all datasets."""
        result = []
        for d in self.datasets:
            result.extend(d.grids)
        return tuple(result)

    def tree(self) -> Node:
        """Generates a hierarchical tree structure for the Grids object.

        Returns
        -------
        Node
            A Node object representing the Grids object.
        """
        return Node(self, [d.tree() for d in self.datasets], mode="concat")

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns:
        Dict[str, Any]: The metadata specific to the forwards subclass.
        """
        return {}


class Cutout(GridsBase):
    """A class to handle hierarchical management of Limited Area Models (LAMs) and a global dataset."""

    def __init__(
        self,
        datasets: List[Any],
        axis: int = 3,
        cropping_distance: float = 2.0,
        neighbours: int = 5,
        min_distance_km: Optional[float] = None,
        plot: Optional[bool] = None,
    ) -> None:
        """Initializes a Cutout object for hierarchical management of Limited Area
        Models (LAMs) and a global dataset, handling overlapping regions.

        Parameters
        ----------
        datasets : list
            List of LAM and global datasets.
        axis : int
            Concatenation axis, must be set to 3.
        cropping_distance : float
            Distance threshold in degrees for cropping cutouts.
        neighbours : int
            Number of neighboring points to consider when constructing masks.
        min_distance_km : float, optional
            Minimum distance threshold in km between grid points.
        plot : bool, optional
            Flag to enable or disable visualization plots.
        """
        super().__init__(datasets, axis)
        assert len(datasets) >= 2, "CutoutGrids requires at least two datasets"
        assert axis == 3, "CutoutGrids requires axis=3"
        assert cropping_distance >= 0, "cropping_distance must be a non-negative number"
        if min_distance_km is not None:
            assert min_distance_km >= 0, "min_distance_km must be a non-negative number"

        self.lams = datasets[:-1]  # Assume the last dataset is the global one
        self.globe = datasets[-1]
        self.axis = axis
        self.cropping_distance = cropping_distance
        self.neighbours = neighbours
        self.min_distance_km = min_distance_km
        self._plot = plot
        self.masks = []  # To store the masks for each LAM dataset
        self.global_mask = np.ones(self.globe.shape[-1], dtype=bool)

        # Initialize cumulative masks
        self._initialize_masks()

    def _initialize_masks(self) -> None:
        """Generate hierarchical masks for each LAM dataset by excluding overlapping regions with previous LAMs and creating a global mask for the global dataset.

        Raises
        ------
        ValueError
            If the global mask dimension does not match the global dataset grid points.
        """
        from anemoi.datasets.grids import cutout_mask

        for i, lam in enumerate(self.lams):
            assert len(lam.shape) == len(
                self.globe.shape
            ), "LAMs and global dataset must have the same number of dimensions"
            lam_lats = lam.latitudes
            lam_lons = lam.longitudes
            # Create a mask for the global dataset excluding all LAM points
            global_overlap_mask = cutout_mask(
                lam.latitudes,
                lam.longitudes,
                self.globe.latitudes,
                self.globe.longitudes,
                plot=self._plot,
                min_distance_km=self.min_distance_km,
                cropping_distance=self.cropping_distance,
                neighbours=self.neighbours,
            )

            # Ensure the mask dimensions match the global grid points
            if global_overlap_mask.shape[0] != self.globe.shape[-1]:
                raise ValueError("Global mask dimension does not match global dataset grid " "points.")
            self.global_mask[~global_overlap_mask] = False

            # Create a mask for the LAM datasets hierarchically, excluding
            # points from previous LAMs
            lam_current_mask = np.ones(lam.shape[-1], dtype=bool)
            if i > 0:
                for j in range(i):
                    prev_lam = self.lams[j]
                    prev_lam_lats = prev_lam.latitudes
                    prev_lam_lons = prev_lam.longitudes
                    # Check for overlap by computing distances
                    if self.has_overlap(prev_lam_lats, prev_lam_lons, lam_lats, lam_lons):
                        lam_overlap_mask = cutout_mask(
                            prev_lam_lats,
                            prev_lam_lons,
                            lam_lats,
                            lam_lons,
                            plot=self._plot,
                            min_distance_km=self.min_distance_km,
                            cropping_distance=self.cropping_distance,
                            neighbours=self.neighbours,
                        )
                        lam_current_mask[~lam_overlap_mask] = False
            self.masks.append(lam_current_mask)

    def has_overlap(
        self,
        lats1: NDArray[Any],
        lons1: NDArray[Any],
        lats2: NDArray[Any],
        lons2: NDArray[Any],
        distance_threshold: float = 1.0,
    ) -> bool:
        """Check for overlapping points between two sets of latitudes and longitudes within a specified distance threshold.

        Parameters
        ----------
        lats1 : NDArray[Any]
            Latitude array for the first dataset.
        lons1 : NDArray[Any]
            Longitude array for the first dataset.
        lats2 : NDArray[Any]
            Latitude array for the second dataset.
        lons2 : NDArray[Any]
            Longitude array for the second dataset.
        distance_threshold : float
            Distance in degrees to consider as overlapping.

        Returns
        -------
        bool
            True if any points overlap within the distance threshold, otherwise False.
        """
        # Create KDTree for the first set of points
        tree = cKDTree(np.vstack((lats1, lons1)).T)

        # Query the second set of points against the first tree
        distances, _ = tree.query(np.vstack((lats2, lons2)).T, k=1)

        # Check if any distance is less than the specified threshold
        return np.any(distances < distance_threshold)

    def __getitem__(self, index: FullIndex) -> NDArray[Any]:
        """Retrieve data from the masked LAMs and global dataset based on the given index.

        Parameters
        ----------
        index : FullIndex
            Index specifying the data to retrieve.

        Returns
        -------
        NDArray[Any]
            Data array from the masked datasets based on the index.
        """
        if isinstance(index, (int, slice)):
            index = (index, slice(None), slice(None), slice(None))
        return self._get_tuple(index)

    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Helper method that applies masks and retrieves data from each dataset according to the specified index.

        Parameters
        ----------
        index : TupleIndex
            Index specifying slices to retrieve data.

        Returns
        -------
        NDArray[Any]
            Concatenated data array from all datasets based on the index.
        """
        index, changes = index_to_slices(index, self.shape)
        # Select data from each LAM
        lam_data = [lam[index[:3]] for lam in self.lams]

        # First apply spatial indexing on `self.globe` and then apply the mask
        globe_data_sliced = self.globe[index[:3]]
        globe_data = globe_data_sliced[..., self.global_mask]

        # Concatenate LAM data with global data, apply the grid slicing
        result = np.concatenate(lam_data + [globe_data], axis=self.axis)[..., index[3]]

        return apply_index_to_slices_changes(result, changes)

    def collect_supporting_arrays(self, collected: List[Any], *path: Any) -> None:
        """Collect supporting arrays, including masks for each LAM and the global dataset.

        Parameters
        ----------
        collected : List[Any]
            List to which the supporting arrays are appended.
        *path : Any
            Variable length argument list specifying the paths for the masks.
        """
        # Append masks for each LAM
        for i, (lam, mask) in enumerate(zip(self.lams, self.masks)):
            collected.append((path + (f"lam_{i}",), "cutout_mask", mask))

        # Append the global mask
        collected.append((path + ("global",), "cutout_mask", self.global_mask))

    @cached_property
    def shape(self) -> Shape:
        """Returns the shape of the Cutout, accounting for retained grid points
        across all LAMs and the global dataset.
        """
        shapes = [np.sum(mask) for mask in self.masks]
        global_shape = np.sum(self.global_mask)
        total_shape = sum(shapes) + global_shape
        return tuple(self.lams[0].shape[:-1] + (int(total_shape),))

    def check_same_resolution(self, d1: Dataset, d2: Dataset) -> None:
        """Checks if the resolutions of two datasets are the same.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        # Turned off because we are combining different resolutions
        pass

    @property
    def grids(self) -> TupleIndex:
        """Returns the number of grid points for each LAM and the global dataset
        after applying masks.
        """
        grids = [np.sum(mask) for mask in self.masks]
        grids.append(np.sum(self.global_mask))
        return tuple(grids)

    @property
    def latitudes(self) -> NDArray[Any]:
        """Returns the concatenated latitudes of each LAM and the global dataset
        after applying masks.
        """
        lam_latitudes = np.concatenate([lam.latitudes[mask] for lam, mask in zip(self.lams, self.masks)])

        assert (
            len(lam_latitudes) + len(self.globe.latitudes[self.global_mask]) == self.shape[-1]
        ), "Mismatch in number of latitudes"

        latitudes = np.concatenate([lam_latitudes, self.globe.latitudes[self.global_mask]])
        return latitudes

    @property
    def longitudes(self) -> NDArray[Any]:
        """Returns the concatenated longitudes of each LAM and the global dataset
        after applying masks.
        """
        lam_longitudes = np.concatenate([lam.longitudes[mask] for lam, mask in zip(self.lams, self.masks)])

        assert (
            len(lam_longitudes) + len(self.globe.longitudes[self.global_mask]) == self.shape[-1]
        ), "Mismatch in number of longitudes"

        longitudes = np.concatenate([lam_longitudes, self.globe.longitudes[self.global_mask]])
        return longitudes

    def tree(self) -> Node:
        """Generates a hierarchical tree structure for the `Cutout` instance and
        its associated datasets.

        Returns
        -------
        Node
            A `Node` object representing the `Cutout` instance as the root
            node, with each dataset in `self.datasets` represented as a child
            node.
        """
        return Node(self, [d.tree() for d in self.datasets])

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Returns metadata specific to the Cutout object.

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the Cutout object.
        """
        return {}


def grids_factory(args: Tuple[Any, ...], kwargs: dict) -> Dataset:
    """Factory function to create a Grids object.

    Parameters
    ----------
    args : Tuple[Any, ...]
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    Dataset
        A Grids object.
    """
    if "ensemble" in kwargs:
        raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")

    grids = kwargs.pop("grids")
    axis = kwargs.pop("axis", 3)

    assert len(args) == 0
    assert isinstance(grids, (list, tuple))

    datasets = [_open(e) for e in grids]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Grids(datasets, axis=axis)._subset(**kwargs)


def cutout_factory(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dataset:
    """Factory function to create a Cutout object.

    Parameters
    ----------
    args : Tuple[Any, ...]
        Positional arguments.
    kwargs : Dict[str, Any]
        Keyword arguments.

    Returns
    -------
    Dataset
        A Cutout object.
    """
    if "ensemble" in kwargs:
        raise NotImplementedError("Cannot use both 'ensemble' and 'cutout'")

    cutout = kwargs.pop("cutout")
    axis = kwargs.pop("axis", 3)
    plot = kwargs.pop("plot", None)
    min_distance_km = kwargs.pop("min_distance_km", None)
    cropping_distance = kwargs.pop("cropping_distance", 2.0)
    neighbours = kwargs.pop("neighbours", 5)

    assert len(args) == 0
    assert isinstance(cutout, (list, tuple)), "cutout must be a list or tuple"

    datasets = [_open(e) for e in cutout]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Cutout(
        datasets,
        axis=axis,
        neighbours=neighbours,
        min_distance_km=min_distance_km,
        cropping_distance=cropping_distance,
        plot=plot,
    )._subset(**kwargs)
