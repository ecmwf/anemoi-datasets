# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest

from anemoi.datasets.grids import cutout_mask


def test_cutout_mask_with_max_distance():
    """Test cutout_mask with max_distance_km parameter."""
    # Create a LAM region
    lam_lat_range = np.linspace(44.0, 46.0, 10)
    lam_lon_range = np.linspace(0.0, 2.0, 10)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    # Create a global grid with points at varying distances
    global_lats = np.array([44.0, 45.0, 45.5, 46.0, 50.0])
    global_lons = np.array([359.0, 0.0, 1.0, 2.0, 0.0])

    # Apply mask with max_distance_km to exclude far points
    mask = cutout_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        max_distance_km=100.0,  # 100 km limit
    )

    # The last point at lat=50.0 should be excluded (too far)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == global_lats.shape
    assert np.array_equal(mask, np.array([False, False, False, False, True]))


def test_cutout_mask_with_min_distance():
    """Test cutout_mask with both min_distance_km."""
    # Create a LAM region
    lam_lat_range = np.linspace(44.0, 46.0, 10)
    lam_lon_range = np.linspace(0.0, 2.0, 10)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    # Create a global grid
    global_lats = np.array([45.0, 45.5, 46.0, 46.1])
    global_lons = np.array([0.0, 1.0, 2.0, 0.1])

    mask = cutout_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        min_distance_km=100.0,
    )

    # The last point at lat=50.0 should be excluded (too close)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == global_lats.shape
    assert np.array_equal(mask, np.array([False, False, False, False, True]))


def test_cutout_mask_array_shapes():
    """Test that input arrays must be 1D."""
    lam_lats = np.array([[45.0, 45.0], [46.0, 46.0]])
    lam_lons = np.array([[0.0, 1.0], [0.0, 1.0]])
    global_lats = np.array([45.0])
    global_lons = np.array([0.0])

    # Should raise assertion error due to 2D arrays
    with pytest.raises(AssertionError):
        cutout_mask(lam_lats, lam_lons, global_lats, global_lons)


def test_cutout_mask_parameter_types():
    """Test that max_distance_km accepts int and float."""
    lam_lat_range = np.linspace(44.0, 46.0, 10)
    lam_lon_range = np.linspace(0.0, 2.0, 10)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    global_lats = np.array([45.0, 46.0])
    global_lons = np.array([0.0, 2.0])

    # Test with int
    mask_int = cutout_mask(
        lam_lats, lam_lons, global_lats, global_lons, max_distance_km=100
    )
    assert isinstance(mask_int, np.ndarray)

    # Test with float
    mask_float = cutout_mask(
        lam_lats, lam_lons, global_lats, global_lons, max_distance_km=100.0
    )
    assert isinstance(mask_float, np.ndarray)


def test_cutout_mask_large_grid():
    """Test cutout_mask with a larger, more realistic grid."""
    # Create a LAM region (10x10 grid)
    lam_lat_range = np.linspace(40.0, 50.0, 10)
    lam_lon_range = np.linspace(0.0, 10.0, 10)
    lam_lats, lam_lons = np.meshgrid(lam_lat_range, lam_lon_range)
    lam_lats = lam_lats.flatten()
    lam_lons = lam_lons.flatten()

    # Create a global grid (20x20 grid)
    global_lat_range = np.linspace(30.0, 60.0, 20)
    global_lon_range = np.linspace(-10.0, 20.0, 20)
    global_lats, global_lons = np.meshgrid(global_lat_range, global_lon_range)
    global_lats = global_lats.flatten()
    global_lons = global_lons.flatten()

    mask = cutout_mask(
        lam_lats,
        lam_lons,
        global_lats,
        global_lons,
        min_distance_km=50.0,
        max_distance_km=300.0,
    )

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (400,)  # 20x20 flattened
    assert mask.dtype == bool
    # Some points should be masked (excluded)
    assert np.any(mask)
    # Some points should not be masked
    assert not np.all(mask)
