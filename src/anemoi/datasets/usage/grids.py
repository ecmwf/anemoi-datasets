# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import numpy as np
from anemoi.utils.grids import latlon_to_xyz
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


def plot_mask(
    path: str,
    mask: NDArray[Any],
    lats: NDArray[Any],
    lons: NDArray[Any],
    global_lats: NDArray[Any],
    global_lons: NDArray[Any],
) -> None:
    """Plot and save various visualizations of the mask and coordinates.

    Parameters
    ----------
    path : str
        The base path for saving the plots.
    mask : NDArray[Any]
        The mask array.
    lats : NDArray[Any]
        Latitude coordinates.
    lons : NDArray[Any]
        Longitude coordinates.
    global_lats : NDArray[Any]
        Global latitude coordinates.
    global_lons : NDArray[Any]
        Global longitude coordinates.
    """
    import matplotlib.pyplot as plt

    s = 1

    global_lons[global_lons >= 180] -= 360

    plt.figure(figsize=(10, 5))
    plt.scatter(global_lons, global_lats, s=s, marker="o", c="r")
    if isinstance(path, str):
        plt.savefig(path + "-global.png")

    plt.figure(figsize=(10, 5))
    plt.scatter(global_lons[mask], global_lats[mask], s=s, c="k")
    if isinstance(path, str):
        plt.savefig(path + "-cutout.png")

    plt.figure(figsize=(10, 5))
    plt.scatter(lons, lats, s=s)
    if isinstance(path, str):
        plt.savefig(path + "-lam.png")
    # plt.scatter(lons, lats, s=0.01)

    plt.figure(figsize=(10, 5))
    plt.scatter(global_lons[mask], global_lats[mask], s=s, c="r")
    plt.scatter(lons, lats, s=s)
    if isinstance(path, str):
        plt.savefig(path + "-both.png")
    # plt.scatter(lons, lats, s=0.01)

    plt.figure(figsize=(10, 5))
    plt.scatter(global_lons[mask], global_lats[mask], s=s, c="r")
    plt.scatter(lons, lats, s=s)
    plt.xlim(np.amin(lons) - 1, np.amax(lons) + 1)
    plt.ylim(np.amin(lats) - 1, np.amax(lats) + 1)
    if isinstance(path, str):
        plt.savefig(path + "-both-zoomed.png")

    plt.figure(figsize=(10, 5))
    plt.scatter(global_lons[mask], global_lats[mask], s=s, c="r")
    plt.xlim(np.amin(lons) - 1, np.amax(lons) + 1)
    plt.ylim(np.amin(lats) - 1, np.amax(lats) + 1)
    if isinstance(path, str):
        plt.savefig(path + "-global-zoomed.png")


class Triangle3D:
    """A class to represent a 3D triangle and perform intersection tests with rays."""

    def __init__(self, v0: NDArray[Any], v1: NDArray[Any], v2: NDArray[Any]) -> None:
        """Initialize the Triangle3D object.

        Parameters
        ----------
        v0 : NDArray[Any]
            First vertex of the triangle.
        v1 : NDArray[Any]
            Second vertex of the triangle.
        v2 : NDArray[Any]
            Third vertex of the triangle.
        """
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def intersect(self, ray_origin: NDArray[Any], ray_direction: NDArray[Any]) -> bool:
        """Check if a ray intersects with the triangle.

        Parameters
        ----------
        ray_origin : NDArray[Any]
            Origin of the ray.
        ray_direction : NDArray[Any]
            Direction of the ray.

        Returns
        -------
        bool
            True if the ray intersects with the triangle, False otherwise.
        """
        # Möller–Trumbore intersection algorithm
        # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

        epsilon = 0.0000001

        h = np.cross(ray_direction, self.v2 - self.v0)
        a = np.dot(self.v1 - self.v0, h)

        if -epsilon < a < epsilon:
            return False

        f = 1.0 / a
        s = ray_origin - self.v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return False

        q = np.cross(s, self.v1 - self.v0)
        v = f * np.dot(ray_direction, q)

        if v < 0.0 or u + v > 1.0:
            return False

        t = f * np.dot(self.v2 - self.v0, q)

        if t > epsilon:
            return True

        return False


def cropping_mask(
    lats: NDArray[Any],
    lons: NDArray[Any],
    north: float,
    west: float,
    south: float,
    east: float,
) -> NDArray[Any]:
    """Create a mask for the points within the specified latitude and longitude bounds.

    Parameters
    ----------
    lats : NDArray[Any]
        Latitude coordinates.
    lons : NDArray[Any]
        Longitude coordinates.
    north : float
        Northern boundary.
    west : float
        Western boundary.
    south : float
        Southern boundary.
    east : float
        Eastern boundary.

    Returns
    -------
    NDArray[Any]
        Mask array.
    """
    mask = (
        (lats >= south)
        & (lats <= north)
        & (
            ((lons >= west) & (lons <= east))
            | ((lons >= west + 360) & (lons <= east + 360))
            | ((lons >= west - 360) & (lons <= east - 360))
        )
    )
    return mask


def cutout_mask(
    lats: NDArray[Any],
    lons: NDArray[Any],
    global_lats: NDArray[Any],
    global_lons: NDArray[Any],
    cropping_distance: float = 2.0,
    neighbours: int = 5,
    min_distance_km: int | float | None = None,
    plot: str | None = None,
) -> NDArray[Any]:
    """Return a mask for the points in [global_lats, global_lons] that are inside of [lats, lons].

    Parameters
    ----------
    lats : NDArray[Any]
        Latitude coordinates.
    lons : NDArray[Any]
        Longitude coordinates.
    global_lats : NDArray[Any]
        Global latitude coordinates.
    global_lons : NDArray[Any]
        Global longitude coordinates.
    cropping_distance : float, optional
        Cropping distance. Defaults to 2.0.
    neighbours : int, optional
        Number of neighbours. Defaults to 5.
    min_distance_km : Optional[Union[int, float]], optional
        Minimum distance in kilometers. Defaults to None.
    plot : Optional[str], optional
        Path for saving the plot. Defaults to None.

    Returns
    -------
    NDArray[Any]
        Mask array.
    """
    from scipy.spatial import cKDTree

    # TODO: transform min_distance from lat/lon to xyz

    assert global_lats.ndim == 1
    assert global_lons.ndim == 1
    assert lats.ndim == 1
    assert lons.ndim == 1

    assert global_lats.shape == global_lons.shape
    assert lats.shape == lons.shape

    north = np.amax(lats)
    south = np.amin(lats)
    east = np.amax(lons)
    west = np.amin(lons)

    # Reduce the global grid to the area of interest

    mask = cropping_mask(
        global_lats,
        global_lons,
        np.min([90.0, north + cropping_distance]),
        west - cropping_distance,
        np.max([-90.0, south - cropping_distance]),
        east + cropping_distance,
    )

    # return mask
    # mask = np.array([True] * len(global_lats), dtype=bool)
    global_lats_masked = global_lats[mask]
    global_lons_masked = global_lons[mask]

    global_xyx = latlon_to_xyz(global_lats_masked, global_lons_masked)
    global_points = np.array(global_xyx).transpose()

    xyx = latlon_to_xyz(lats, lons)
    lam_points = np.array(xyx).transpose()

    if isinstance(min_distance_km, (int, float)):
        min_distance = min_distance_km / 6371.0
    else:
        points = {"lam": lam_points, "global": global_points, None: global_points}[min_distance_km]
        distances, _ = cKDTree(points).query(points, k=2)
        min_distance = np.min(distances[:, 1])

        LOG.info(f"cutout_mask using min_distance = {min_distance * 6371.0} km")

    # Use a cKDTree to find the nearest points
    distances, indices = cKDTree(lam_points).query(global_points, k=neighbours)

    # Centre of the Earth
    zero = np.array([0.0, 0.0, 0.0])

    # After the loop, 'inside_lam' will contain a list point to EXCLUDE
    inside_lam = []

    for i, (global_point, distance, index) in enumerate(zip(global_points, distances, indices)):

        # We check more than one triangle in case te global point
        # is near the edge of triangle, (the lam point and global points are colinear)

        inside = False
        for j in range(neighbours):
            t = Triangle3D(
                lam_points[index[j]], lam_points[index[(j + 1) % neighbours]], lam_points[index[(j + 2) % neighbours]]
            )
            inside = t.intersect(zero, global_point)
            if inside:
                break

        close = np.min(distance) <= min_distance

        inside_lam.append(inside or close)

    j = 0
    inside_lam_array = np.array(inside_lam)
    for i, m in enumerate(mask):
        if not m:
            continue

        mask[i] = inside_lam_array[j]
        j += 1

    assert j == len(inside_lam_array)

    # Invert the mask, so we have only the points outside the cutout
    mask = ~mask

    if plot:
        plot_mask(plot, mask, lats, lons, global_lats, global_lons)

    return mask


def thinning_mask(
    lats: NDArray[Any],
    lons: NDArray[Any],
    global_lats: NDArray[Any],
    global_lons: NDArray[Any],
    cropping_distance: float = 2.0,
) -> NDArray[Any]:
    """Return the list of points in [lats, lons] closest to [global_lats, global_lons].

    Parameters
    ----------
    lats : NDArray[Any]
        Latitude coordinates.
    lons : NDArray[Any]
        Longitude coordinates.
    global_lats : NDArray[Any]
        Global latitude coordinates.
    global_lons : NDArray[Any]
        Global longitude coordinates.
    cropping_distance : float, optional
        Cropping distance. Defaults to 2.0.

    Returns
    -------
    NDArray[Any]
        Array of indices of the closest points.
    """
    from scipy.spatial import cKDTree

    assert global_lats.ndim == 1
    assert global_lons.ndim == 1
    assert lats.ndim == 1
    assert lons.ndim == 1

    assert global_lats.shape == global_lons.shape
    assert lats.shape == lons.shape

    north = np.amax(lats)
    south = np.amin(lats)
    east = np.amax(lons)
    west = np.amin(lons)

    # Reduce the global grid to the area of interest

    mask = cropping_mask(
        global_lats,
        global_lons,
        np.min([90.0, north + cropping_distance]),
        west - cropping_distance,
        np.max([-90.0, south - cropping_distance]),
        east + cropping_distance,
    )

    # return mask
    global_lats_masked = global_lats[mask]
    global_lons_masked = global_lons[mask]

    global_xyx = latlon_to_xyz(global_lats_masked, global_lons_masked)
    global_points = np.array(global_xyx).transpose()

    xyx = latlon_to_xyz(lats, lons)
    points = np.array(xyx).transpose()

    # Use a cKDTree to find the nearest points
    _, indices = cKDTree(points).query(global_points, k=1)

    return np.array([i for i in indices])


def outline(lats: NDArray[Any], lons: NDArray[Any], neighbours: int = 5) -> list[int]:
    """Find the outline of the grid points.

    Parameters
    ----------
    lats : NDArray[Any]
        Latitude coordinates.
    lons : NDArray[Any]
        Longitude coordinates.
    neighbours : int, optional
        Number of neighbours. Defaults to 5.

    Returns
    -------
    List[int]
        Indices of the outline points.
    """
    from scipy.spatial import cKDTree

    xyx = latlon_to_xyz(lats, lons)
    grid_points = np.array(xyx).transpose()

    # Use a cKDTree to find the nearest points
    _, indices = cKDTree(grid_points).query(grid_points, k=neighbours)

    # Centre of the Earth
    zero = np.array([0.0, 0.0, 0.0])

    outside = []

    for i, (point, index) in enumerate(zip(grid_points, indices)):
        inside = False
        for j in range(1, neighbours):
            t = Triangle3D(
                grid_points[index[j]],
                grid_points[index[(j + 1) % neighbours]],
                grid_points[index[(j + 2) % neighbours]],
            )
            inside = t.intersect(zero, point)
            if inside:
                break

        if not inside:
            outside.append(i)

    return outside


def nearest_grid_points(
    source_latitudes: NDArray[Any],
    source_longitudes: NDArray[Any],
    target_latitudes: NDArray[Any],
    target_longitudes: NDArray[Any],
    max_distance: float = None,
    k: int = 1,
) -> NDArray[Any]:
    """Find the nearest grid points from source to target coordinates.

    Parameters
    ----------
    source_latitudes : NDArray[Any]
        Source latitude coordinates.
    source_longitudes : NDArray[Any]
        Source longitude coordinates.
    target_latitudes : NDArray[Any]
        Target latitude coordinates.
    target_longitudes : NDArray[Any]
        Target longitude coordinates.
    max_distance: float, optional
        Maximum distance between nearest point and point to interpolate. Defaults to None.
        For example, 1e-3 is 1 km.
    k : int, optional
        The number of k closest neighbors to consider for interpolation

    Returns
    -------
    NDArray[Any]
        Indices of the nearest grid points.
    """
    # TODO: Use the one from anemoi.utils.grids instead
    # from anemoi.utils.grids import ...
    from scipy.spatial import cKDTree

    source_xyz = latlon_to_xyz(source_latitudes, source_longitudes)
    source_points = np.array(source_xyz).transpose()

    target_xyz = latlon_to_xyz(target_latitudes, target_longitudes)
    target_points = np.array(target_xyz).transpose()
    if max_distance is None:
        distances, indices = cKDTree(source_points).query(target_points, k=k)
    else:
        distances, indices = cKDTree(source_points).query(target_points, k=k, distance_upper_bound=max_distance)
    return distances, indices
