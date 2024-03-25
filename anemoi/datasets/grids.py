# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np


def plot_mask(path, mask, lats, lons, global_lats, global_lons):
    import matplotlib.pyplot as plt

    middle = (np.amin(lons) + np.amax(lons)) / 2
    print("middle", middle)
    s = 1

    # gmiddle = (np.amin(global_lons)+ np.amax(global_lons))/2

    # print('gmiddle', gmiddle)
    # global_lons = global_lons-gmiddle+middle
    global_lons[global_lons >= 180] -= 360

    plt.figure(figsize=(10, 5))
    plt.scatter(global_lons, global_lats, s=s, marker="o", c="r")
    plt.savefig(path + "-global.png")

    plt.figure(figsize=(10, 5))
    plt.scatter(global_lons[mask], global_lats[mask], s=s, c="k")
    plt.savefig(path + "-cutout.png")

    plt.figure(figsize=(10, 5))
    plt.scatter(lons, lats, s=s)
    plt.savefig(path + "-lam.png")
    # plt.scatter(lons, lats, s=0.01)

    plt.figure(figsize=(10, 5))
    plt.scatter(global_lons[mask], global_lats[mask], s=s, c="r")
    plt.scatter(lons, lats, s=s)
    plt.savefig(path + "-both.png")
    # plt.scatter(lons, lats, s=0.01)


def latlon_to_xyz(lat, lon, radius=1.0):
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    # We assume that the Earth is a sphere of radius 1 so N(phi) = 1
    # We assume h = 0
    #
    phi = np.deg2rad(lat)
    lda = np.deg2rad(lon)

    cos_phi = np.cos(phi)
    cos_lda = np.cos(lda)
    sin_phi = np.sin(phi)
    sin_lda = np.sin(lda)

    x = cos_phi * cos_lda * radius
    y = cos_phi * sin_lda * radius
    z = sin_phi * radius

    return x, y, z


class Triangle3D:
    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    def intersect(self, ray_origin, ray_direction):
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


def cropping_mask(lats, lons, north, west, south, east):
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
    lats,
    lons,
    global_lats,
    global_lons,
    cropping_distance=2.0,
    min_distance_km=0.0,
    plot=None,
):
    """
    Return a mask for the points in [global_lats, global_lons] that are inside of [lats, lons]
    """
    from scipy.spatial import KDTree

    # TODO: transform min_distance from lat/lon to xyz

    min_distance = min_distance_km / 6371.0

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

    # Use a KDTree to find the nearest points
    kdtree = KDTree(lam_points)
    distances, indices = kdtree.query(global_points, k=3)

    zero = np.array([0.0, 0.0, 0.0])
    ok = []
    for i, (global_point, distance, index) in enumerate(zip(global_points, distances, indices)):
        t = Triangle3D(lam_points[index[0]], lam_points[index[1]], lam_points[index[2]])
        # distance = np.min(distance)
        # The point is inside the triangle if the intersection with the ray
        # from the point to the center of the Earth is not None
        # (the direction of the ray is not important)

        intersect = t.intersect(zero, global_point) or t.intersect(global_point, zero)
        close = np.min(distance) <= min_distance

        ok.append(intersect and not close)

    j = 0
    ok = np.array(ok)
    for i, m in enumerate(mask):
        if not m:
            continue

        mask[i] = ok[j]
        j += 1

    assert j == len(ok)

    # Invert the mask, so we have only the points outside the cutout
    mask = ~mask

    if plot:
        plot_mask(plot, mask, lats, lons, global_lats, global_lons)

    return mask


def thinning_mask(
    lats,
    lons,
    global_lats,
    global_lons,
    cropping_distance=2.0,
):
    """
    Return the list of points in [lats, lons] closest to [global_lats, global_lons]
    """
    from scipy.spatial import KDTree

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

    # Use a KDTree to find the nearest points
    kdtree = KDTree(points)
    _, indices = kdtree.query(global_points, k=1)

    return np.array([i for i in indices])


if __name__ == "__main__":
    global_lats, global_lons = np.meshgrid(
        np.linspace(90, -90, 90),
        np.linspace(-180, 180, 180),
    )
    global_lats = global_lats.flatten()
    global_lons = global_lons.flatten()

    lats, lons = np.meshgrid(
        np.linspace(50, 40, 100),
        np.linspace(-10, 15, 100),
    )
    lats = lats.flatten()
    lons = lons.flatten()

    mask = cutout_mask(lats, lons, global_lats, global_lons, cropping_distance=5.0)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 5))
    plt.scatter(global_lons, global_lats, s=0.01, marker="o", c="r")
    plt.scatter(global_lons[mask], global_lats[mask], s=0.1, c="k")
    # plt.scatter(lons, lats, s=0.01)
    plt.savefig("cutout.png")
