# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import Any

import numpy as np

from ..debug import Node

LOG = logging.getLogger(__name__)


class _Thinner(ABC):

    @abstractmethod
    def mask(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def _lat_lon_to_xyz(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        from anemoi.utils.grids import latlon_to_xyz

        xyz = latlon_to_xyz(latitudes, longitudes)
        return np.array(xyz).transpose()


class _EveryNth(_Thinner):
    def __init__(self, thinning: float | int, field_shape: tuple[int, ...] | None):

        if field_shape is None:
            raise ValueError("Field shape must be provided for every-nth thinning method")

        if len(field_shape) != 2:
            raise ValueError("Thinning only works latitude/longitude fields")

        self.thinning = thinning
        self.field_shape = field_shape

    def mask(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        """Create a mask to thin the points by keeping every N-th point in both latitude and longitude."""

        latitudes = latitudes.reshape(self.field_shape)
        longitudes = longitudes.reshape(self.field_shape)
        latitudes = latitudes[:: self.thinning, :: self.thinning].flatten()
        longitudes = longitudes[:: self.thinning, :: self.thinning].flatten()

        # TODO: This is not very efficient

        mask = [lat in latitudes and lon in longitudes for lat, lon in zip(latitudes, longitudes)]
        mask = np.array(mask, dtype=bool)
        return mask


class _DistanceBased(_Thinner):
    def __init__(self, thinning: float | int, field_shape: tuple[int, ...] | None = None):
        self.min_distance = thinning / 6371.0  # Convert from km

    def mask(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:

        from scipy.spatial import KDTree

        points = self._lat_lon_to_xyz(latitudes, longitudes)

        tree = KDTree(points)
        mask = np.ones(points.shape[0], dtype=bool)

        # For each point, remove points within the min_distance radius
        for i, point in enumerate(points):
            if mask[i]:
                indices = tree.query_ball_point(point, r=self.min_distance)
                mask[indices] = False
                mask[i] = True  # Keep the current point

        return mask


class _GridThinning(_Thinner):
    def __init__(self, thinning: float | int, field_shape: tuple[int, ...] | None = None):
        self.grid_size = thinning / 6371.0  # Convert km

    def mask(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        """Quantize the points based on grid size"""
        points = self._lat_lon_to_xyz(latitudes, longitudes)

        quantized_points = np.round(points / self.grid_size).astype(int)

        # Use a set to keep unique grid cells (which automatically filters points)
        _, indices = np.unique(quantized_points, axis=0, return_index=True)

        return indices


class _RandomThinning(_Thinner):
    def __init__(self, thinning: float | int, field_shape: tuple[int, ...] | None = None):
        self.fraction = thinning

    def mask(self, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
        """Assume `points` is an NxD array, where N is the number of points, and D is the dimension (e.g., 3D)
        Create a random mask to keep a fraction of the points
        """

        points = self._lat_lon_to_xyz(latitudes, longitudes)
        mask = np.random.rand(points.shape[0]) < self.fraction
        return mask


class ThinningMixin:

    @cached_property
    def thinner(self) -> _Thinner:
        THINNERS = {
            "every-nth": _EveryNth,
            "distance-based": _DistanceBased,
            "grid": _GridThinning,
            "random": _RandomThinning,
        }

        if self.method not in THINNERS:
            raise ValueError(f"Unknown thinning method: {self.method}. Supported methods are: {list(THINNERS.keys())}")

        return THINNERS[self.method](self.thinning, self.grid_shape)

    # Dataset methods below

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], thinning=self.thinning, method=self.method)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the Thinning subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the Thinning subclass.
        """
        return dict(thinning=self.thinning, method=self.method)

    def origin_transformation(self, variable, origins):
        return {
            "name": "thinning",
            "config": dict(thinning=self.thinning, method=self.method),
        }
