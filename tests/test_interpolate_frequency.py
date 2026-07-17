# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for InterpolateFrequency."""

from unittest.mock import patch

import numpy as np
from test_data import create_zarr
from test_data import mockup_open_zarr

from anemoi.datasets import open_dataset


@mockup_open_zarr
def test_interpolate_frequency_preserves_dtype():
    """Interpolated timesteps should have the same dtype as the source data."""
    ds = open_dataset("test-2021-2021-6h-o96-abcd", interpolate_frequency="3h")

    # The mock zarr uses float64; interpolated result should stay float64
    assert ds[0].dtype == np.float64
    assert ds[1].dtype == np.float64


@mockup_open_zarr
def test_interpolate_frequency_doubles_length():
    """Interpolating 6h data to 3h should roughly double the length."""
    ds_orig = open_dataset("test-2021-2021-6h-o96-abcd")
    ds_interp = open_dataset("test-2021-2021-6h-o96-abcd", interpolate_frequency="3h")

    expected_len = (len(ds_orig) - 1) * 2 + 1
    assert len(ds_interp) == expected_len


@mockup_open_zarr
def test_interpolate_frequency_values_correct():
    """Verify interpolation produces correct midpoint values."""
    ds_orig = open_dataset("test-2021-2021-6h-o96-abcd")
    ds_interp = open_dataset("test-2021-2021-6h-o96-abcd", interpolate_frequency="3h")

    # Index 0 in interp == index 0 in original
    np.testing.assert_array_equal(ds_interp[0], ds_orig[0])

    # Index 1 in interp should be midpoint of original[0] and original[1]
    expected = (ds_orig[0] + ds_orig[1]) / 2
    np.testing.assert_allclose(ds_interp[1], expected, rtol=1e-6)

    # Index 2 in interp == index 1 in original
    np.testing.assert_array_equal(ds_interp[2], ds_orig[1])


@mockup_open_zarr
def test_interpolate_frequency_negative_index_preserves_dtype():
    """Negative indexing should also preserve dtype."""
    ds = open_dataset("test-2021-2021-6h-o96-abcd", interpolate_frequency="3h")
    assert ds[-1].dtype == np.float64
    assert ds[-2].dtype == np.float64


def test_interpolate_frequency_preserves_float32():
    """Float32 source data must not be promoted to float64 during interpolation.

    This is a regression test: linear interpolation arithmetic can promote
    float32 to float64 if the result is not explicitly cast back.
    """
    # Use create_zarr from the test infrastructure, then replace data with float32
    root = create_zarr(vars=["a", "b"], start=2021, end=2021)
    float32_data = root["data"][:].astype(np.float32)
    del root["data"]
    root.create_array("data", data=float32_data, chunks=float32_data.shape, compressor=None)

    with patch("zarr.open", lambda name, mode: root):
        with patch("anemoi.datasets.usage.store.dataset_lookup", lambda name, fail: name + ".zarr"):
            ds = open_dataset("test", interpolate_frequency="3h")

    # All timesteps (original and interpolated) must remain float32
    for i in range(min(len(ds), 10)):
        assert ds[i].dtype == np.float32, f"Timestep {i}: dtype is {ds[i].dtype}, expected float32"
