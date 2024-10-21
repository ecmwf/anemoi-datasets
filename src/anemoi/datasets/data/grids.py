# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np

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
    def __len__(self):
        return sum(len(i) for i in self.datasets)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        # print(index, changes)
        lengths = [d.shape[0] for d in self.datasets]
        slices = length_to_slices(index[0], lengths)
        # print("slies", slices)
        result = [d[update_tuple(index, 0, i)[0]] for (d, i) in zip(self.datasets, slices) if i is not None]
        result = np.concatenate(result, axis=0)
        return apply_index_to_slices_changes(result, changes)

    @debug_indexing
    def __getitem__(self, n):
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
    def _get_slice(self, s):
        result = []

        lengths = [d.shape[0] for d in self.datasets]
        slices = length_to_slices(s, lengths)

        result = [d[i] for (d, i) in zip(self.datasets, slices) if i is not None]

        return np.concatenate(result)

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=0)

    def check_same_lengths(self, d1, d2):
        # Turned off because we are concatenating along the first axis
        pass

    def check_same_dates(self, d1, d2):
        # Turned off because we are concatenating along the dates axis
        pass

    @property
    def dates(self):
        return np.concatenate([d.dates for d in self.datasets])

    @property
    def shape(self):
        return (len(self),) + self.datasets[0].shape[1:]

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])


class GridsBase(GivenAxis):
    def __init__(self, datasets, axis):
        super().__init__(datasets, axis)
        # Shape: (dates, variables, ensemble, 1d-values)
        assert len(datasets[0].shape) == 4, "Grids must be 1D for now"

    def check_same_grid(self, d1, d2):
        # We don't check the grid, because we want to be able to combine
        pass

    def check_same_resolution(self, d1, d2):
        # We don't check the resolution, because we want to be able to combine
        pass


class Grids(GridsBase):
    # TODO: select the statistics of the most global grid?
    @property
    def latitudes(self):
        return np.concatenate([d.latitudes for d in self.datasets])

    @property
    def longitudes(self):
        return np.concatenate([d.longitudes for d in self.datasets])

    @property
    def grids(self):
        result = []
        for d in self.datasets:
            result.extend(d.grids)
        return tuple(result)

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets], mode="concat")


class Cutout(GridsBase):
    def __init__(self, datasets, axis, min_distance_km=None, cropping_distance=2.0, neighbours=5, plot=False):
        from anemoi.datasets.grids import cutout_mask

        super().__init__(datasets, axis)
        assert len(datasets) == 2, "CutoutGrids requires two datasets"
        assert axis == 3, "CutoutGrids requires axis=3"
        
        # Assume that all datasets except the last one are LAMs, and the last 
        # one is the global model
        self.lams = datasets[:-1]
        self.globe = datasets[-1]
        
        # Initialize masks
        self.masks = []
        
        # Track already used points in the global dataset
        used_points = np.zeros(self.globe.shape[3], dtype=bool) 
        
        for lam in self.lams:
            current_mask = cutout_mask(
                lam.latitudes,
                lam.longitudes,
                self.globe.latitudes,
                self.globe.longitudes,
                plot=plot,
                min_distance_km=min_distance_km,
                cropping_distance=cropping_distance,
                neighbours=neighbours,
            )
            # Exclude points already covered by previous LAMs
            current_mask[used_points] = False
            self.masks.append(current_mask)
            # Mark the current LAM's points as used
            # This ensures that all points already covered by any LAM (up to the 
            # current one) are marked as True in used_points, so future LAMs or 
            # the global dataset will avoid using these points.
            used_points |= current_mask
        
        assert len(self.masks) == len(self.lams)
        assert len(used_points) == self.globe.shape[3], "Mask length mismatch with global shape"

    @cached_property
    def shape(self):
        lam_shapes = [lam.shape for lam in self.lams]
        globe_shape = self.globe.shape

        # Number of non-zero masked values in the globe dataset for each LAM
        nb_globe = sum(np.count_nonzero(mask) for mask in self.masks)
        # Total number of grid points covered by all the LAM datasets
        lam_total_length = sum(shape[-1] for shape in lam_shapes)

        # Combine the shape information
        return lam_shapes[0][:-1] + (lam_total_length + nb_globe,)

    def check_same_resolution(self, d1, d2):
        # Turned off because we are combining different resolutions
        pass
    
    @property
    def latitudes(self):
        # Concatenate latitudes from all LAMs and the masked global dataset
        lam_latitudes = np.concatenate([lam.latitudes for lam in self.lams])
        globe_latitudes = np.concatenate(
            [self.globe.latitudes[mask] for mask in self.masks]
            )
        return np.concatenate([lam_latitudes, globe_latitudes])

    @property
    def longitudes(self):
        # Concatenate longitudes from all LAMs and the masked global dataset
        lam_longitudes = np.concatenate([lam.longitudes for lam in self.lams])
        globe_longitudes = np.concatenate([self.globe.longitudes[mask] for mask in self.masks])
        return np.concatenate([lam_longitudes, globe_longitudes])

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            index = (index, slice(None), slice(None), slice(None))
        return self._get_tuple(index)
    
    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        assert self.axis >= len(index) or index[self.axis] == slice(
            None
        ), f"No support for selecting a subset of the 1D values {index} ({self.tree()})"
        index, changes = index_to_slices(index, self.shape)

        # In case index_to_slices has changed the last slice
        index, _ = update_tuple(index, self.axis, slice(None))

        # Handle data from LAMs
        lam_data = [lam[index] for lam in self.lams]
        lam_data = np.concatenate(lam_data, axis=self.axis)

        # Handle masked global data for each LAM
        globe_data = self.globe[index]
        globe_data = np.concatenate(
            [globe_data[:, :, :, mask] for mask in self.masks], 
            axis=self.axis
            )

        # Combine LAM data with global data
        result = np.concatenate([lam_data, globe_data], axis=self.axis)

        return apply_index_to_slices_changes(result, changes)
    
    @property
    def grids(self):
        # Ensure each LAM has only one grid, and then return a combined grid layout
        for d in self.lams:
            if len(d.grids) > 1:
                raise NotImplementedError("CutoutGrids does not support multi-grids datasets as inputs")
        lam_total = sum(lam.shape[-1] for lam in self.lams)
        globe_not_masked = self.shape[-1] - lam_total  # Number of points in the global dataset after cutout
        return (lam_total, globe_not_masked)

    def tree(self):
        return Node(self, [d.tree() for d in self.lams] + [self.globe.tree()], mode="concat")

def grids_factory(args, kwargs):
    if "ensemble" in kwargs:
        raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")

    grids = kwargs.pop("grids")
    axis = kwargs.pop("axis", 3)

    assert len(args) == 0
    assert isinstance(grids, (list, tuple))

    datasets = [_open(e) for e in grids]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Grids(datasets, axis=axis)._subset(**kwargs)    

def cutout_factory(args, kwargs):
    if "ensemble" in kwargs:
        raise NotImplementedError("Cannot use both 'ensemble' and 'cutout'")

    cutout = kwargs.pop("cutout")
    axis = kwargs.pop("axis", 3)
    plot = kwargs.pop("plot", None)
    min_distance_km = kwargs.pop("min_distance_km", None)
    cropping_distance = kwargs.pop("cropping_distance", 2.0)
    neighbours = kwargs.pop("neighbours", 5)

    assert len(args) == 0
    assert isinstance(cutout, (list, tuple))

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
