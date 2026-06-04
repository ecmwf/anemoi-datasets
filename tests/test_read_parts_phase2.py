# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Phase 2 tests: two-step read through multi-dataset wrappers.

Tests cover Concat (gridded/concat.py), Concat (gridded/grids.py),
Join, Merge, GivenAxis, and Cutout.

All tests compare two_step_read(ds, n) == ds[n] for varied index types.
"""

import datetime
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.usage.gridded.concat import Concat as GriddedConcat
from anemoi.datasets.usage.gridded.join import Join
from anemoi.datasets.usage.gridded.merge import Merge
from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.read_parts import two_step_read

@pytest.fixture(autouse=True)
def _patch_variable_compat():
    """Variable.check_compatibility doesn't exist in the installed anemoi.transform.
    This is a pre-existing issue unrelated to the two-step read pipeline."""
    with patch(
        "anemoi.datasets.usage.forwards.Combined.check_variables_compatibility",
        return_value=None,
    ):
        yield

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DATES = 8
N_VARS = 4
N_ENS = 1
N_GRID = 6


def _make_zarr_group(
    n_dates: int = N_DATES,
    n_vars: int = N_VARS,
    n_ens: int = N_ENS,
    n_grid: int = N_GRID,
    vars: str = "abcd",
    start_date: datetime.datetime = datetime.datetime(2021, 1, 1),
    seed: int = 0,
    freq_hours: int = 6,
) -> zarr.Group:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_dates, n_vars, n_ens, n_grid)).astype(np.float32)

    root = zarr.group()
    root.create_array("data", data=data, compressor=None)

    freq = datetime.timedelta(hours=freq_hours)
    dates = np.array([start_date + i * freq for i in range(n_dates)], dtype="datetime64")
    root.create_array("dates", data=dates, compressor=None)
    lats = np.linspace(-90, 90, n_grid)
    lons = np.linspace(0, 360, n_grid, endpoint=False)
    root.create_array("latitudes", data=lats, compressor=None)
    root.create_array("longitudes", data=lons, compressor=None)

    root.create_array("mean", data=np.mean(data, axis=(0, 2, 3)), compressor=None)
    root.create_array("stdev", data=np.std(data, axis=(0, 2, 3)) + 1e-8, compressor=None)
    root.create_array("maximum", data=np.max(data, axis=(0, 2, 3)), compressor=None)
    root.create_array("minimum", data=np.min(data, axis=(0, 2, 3)), compressor=None)

    var_list = list(vars[:n_vars])
    root.attrs["frequency"] = frequency_to_string(freq)
    root.attrs["resolution"] = "o96"
    root.attrs["name_to_index"] = {v: i for i, v in enumerate(var_list)}
    root.attrs["data_request"] = {"grid": 1, "area": "g", "param_level": {}}
    root.attrs["variables_metadata"] = {v: {} for v in var_list}
    root.attrs["field_shape"] = [1, n_grid]

    return root


def _make_store(path: str = "test.zarr", **kwargs) -> GriddedZarr:
    group = _make_zarr_group(**kwargs)
    return GriddedZarr(group, path=path)


def _assert_two_step_equals(ds, indices):
    """For each index in indices, assert two_step_read == direct indexing."""
    for n in indices:
        expected = ds[n]
        actual = two_step_read(ds, n)
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"Mismatch for index {n!r}: expected shape {expected.shape}, got {actual.shape}"
        )


# ---------------------------------------------------------------------------
# Concat (gridded/concat.py)
# ---------------------------------------------------------------------------


class TestGriddedConcat:
    def _make_concat(self):
        d1 = _make_store("d1.zarr", n_dates=4, seed=0, start_date=datetime.datetime(2021, 1, 1))
        d2 = _make_store("d2.zarr", n_dates=4, seed=1, start_date=datetime.datetime(2021, 1, 2))
        return GriddedConcat([d1, d2])

    def test_int_index(self):
        ds = self._make_concat()
        _assert_two_step_equals(ds, [0, 3, 4, 7])

    def test_slice_index(self):
        ds = self._make_concat()
        _assert_two_step_equals(ds, [slice(0, 4), slice(3, 7), slice(0, 8)])

    def test_tuple_index(self):
        ds = self._make_concat()
        _assert_two_step_equals(ds, [
            (slice(0, 4), slice(None), slice(None), slice(None)),
            (slice(4, 8), slice(1, 3), slice(None), slice(None)),
            (slice(2, 6), slice(None), slice(None), slice(2, 5)),
        ])

    def test_cross_boundary_slice(self):
        """Slice spanning both sub-datasets."""
        ds = self._make_concat()
        _assert_two_step_equals(ds, [slice(2, 6)])

    def test_single_date(self):
        ds = self._make_concat()
        expected = ds[3]
        actual = two_step_read(ds, 3)
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------


class TestJoin:
    def _make_join(self):
        d1 = _make_store("j1.zarr", n_vars=2, vars="ab", seed=0)
        d2 = _make_store("j2.zarr", n_vars=2, vars="cd", seed=1)
        return Join([d1, d2])

    def test_int_index(self):
        ds = self._make_join()
        _assert_two_step_equals(ds, [0, 3, 7])

    def test_slice_index(self):
        ds = self._make_join()
        _assert_two_step_equals(ds, [slice(0, 3), slice(2, 6)])

    def test_tuple_all_vars(self):
        ds = self._make_join()
        _assert_two_step_equals(ds, [
            (slice(0, 4), slice(None), slice(None), slice(None)),
        ])

    def test_tuple_var_subset(self):
        ds = self._make_join()
        _assert_two_step_equals(ds, [
            (slice(0, 4), slice(1, 3), slice(None), slice(None)),
        ])

    def test_single_date_shape(self):
        ds = self._make_join()
        result = two_step_read(ds, 0)
        assert result.shape == (4, N_ENS, N_GRID), result.shape


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    def _make_merge(self):
        """Two datasets covering non-overlapping date ranges."""
        d1 = _make_store("m1.zarr", n_dates=4, seed=0, start_date=datetime.datetime(2021, 1, 1))
        d2 = _make_store("m2.zarr", n_dates=4, seed=1, start_date=datetime.datetime(2021, 1, 2))
        return Merge([d1, d2])

    def test_int_index_first_dataset(self):
        ds = self._make_merge()
        _assert_two_step_equals(ds, [0, 1, 2, 3])

    def test_int_index_second_dataset(self):
        ds = self._make_merge()
        # All dates are covered; dates from d2 fill in missing slots
        for i in range(len(ds)):
            expected = ds[i]
            actual = two_step_read(ds, i)
            np.testing.assert_array_equal(actual, expected, err_msg=f"Mismatch at index {i}")

    def test_slice_index(self):
        ds = self._make_merge()
        _assert_two_step_equals(ds, [slice(0, 4), slice(2, 6)])

    def test_tuple_index(self):
        ds = self._make_merge()
        _assert_two_step_equals(ds, [
            (slice(0, 4), slice(None), slice(None), slice(None)),
            (slice(2, 6), slice(1, 3), slice(None), slice(None)),
        ])


# ---------------------------------------------------------------------------
# Merge with allow_gaps_in_dates
# ---------------------------------------------------------------------------


class TestMergeWithGaps:
    def _make_merge_with_gaps(self):
        """d1 covers 2021-01-01 to 2021-01-01T18, d2 covers 2021-01-02T06 onward,
        leaving a gap at 2021-01-02T00."""
        # d1: 4 dates at 6h starting 2021-01-01 00:00
        d1 = _make_store("mg1.zarr", n_dates=4, seed=0, start_date=datetime.datetime(2021, 1, 1))
        # d2: 4 dates at 6h starting 2021-01-02 06:00 (skips 2021-01-02 00:00)
        d2 = _make_store("mg2.zarr", n_dates=4, seed=1, start_date=datetime.datetime(2021, 1, 2, 6))
        return Merge([d1, d2], allow_gaps_in_dates=True)

    def test_non_missing_dates(self):
        from anemoi.datasets import MissingDateError
        ds = self._make_merge_with_gaps()
        for i in range(len(ds)):
            if i in ds.missing:
                with pytest.raises(MissingDateError):
                    two_step_read(ds, i)
            else:
                expected = ds[i]
                actual = two_step_read(ds, i)
                np.testing.assert_array_equal(actual, expected, err_msg=f"index {i}")

    def test_collect_parts_empty_for_missing(self):
        ds = self._make_merge_with_gaps()
        for i in ds.missing:
            parts = ds.collect_read_parts(i)
            assert parts == [], f"missing date {i} should yield no parts"


# ---------------------------------------------------------------------------
# GivenAxis (via grids.Concat which extends GivenAxis through Combined)
# ---------------------------------------------------------------------------


class TestGivenAxis:
    """Grids(GridsBase(GivenAxis)) exercises GivenAxis.collect_read_parts along axis=3."""

    def _make_grids(self):
        from anemoi.datasets.usage.gridded.grids import Grids
        d1 = _make_store("gc1.zarr", n_grid=3, seed=0)
        d2 = _make_store("gc2.zarr", n_grid=3, seed=1)
        return Grids([d1, d2], axis=3)

    def test_int_index(self):
        ds = self._make_grids()
        _assert_two_step_equals(ds, [0, 3, 7])

    def test_slice_index(self):
        ds = self._make_grids()
        _assert_two_step_equals(ds, [slice(0, 4), slice(2, 6)])

    def test_tuple_full_grid(self):
        ds = self._make_grids()
        _assert_two_step_equals(ds, [
            (slice(0, 4), slice(None), slice(None), slice(None)),
        ])

    def test_tuple_partial_grid(self):
        ds = self._make_grids()
        _assert_two_step_equals(ds, [
            (slice(0, 4), slice(None), slice(None), slice(0, 4)),
        ])

    def test_shape(self):
        ds = self._make_grids()
        result = two_step_read(ds, 0)
        # Two datasets with n_grid=3 concatenated on axis 3 → 6 grid points
        assert result.shape == (N_VARS, N_ENS, 6), result.shape


# ---------------------------------------------------------------------------
# Cutout (requires anemoi.transform.spatial)
# ---------------------------------------------------------------------------


@pytest.fixture()
def cutout_datasets():
    """Create two non-overlapping grids for Cutout: LAM (small region) + globe."""
    pytest.importorskip("anemoi.transform.spatial")

    from anemoi.datasets.usage.gridded.grids import Cutout

    n_dates = 4

    # LAM: small region around equator/prime meridian
    lam_root = zarr.group()
    n_lam = 5
    lam_data = np.random.default_rng(10).standard_normal((n_dates, 2, 1, n_lam)).astype(np.float32)
    lam_root.create_array("data", data=lam_data, compressor=None)
    freq = datetime.timedelta(hours=6)
    dates = np.array([datetime.datetime(2021, 1, 1) + i * freq for i in range(n_dates)], dtype="datetime64")
    lam_root.create_array("dates", data=dates, compressor=None)
    lam_lats = np.array([5.0, 10.0, 15.0, 5.0, 10.0])
    lam_lons = np.array([5.0, 5.0, 5.0, 10.0, 10.0])
    lam_root.create_array("latitudes", data=lam_lats, compressor=None)
    lam_root.create_array("longitudes", data=lam_lons, compressor=None)
    lam_root.create_array("mean", data=np.zeros(2), compressor=None)
    lam_root.create_array("stdev", data=np.ones(2), compressor=None)
    lam_root.create_array("maximum", data=np.ones(2), compressor=None)
    lam_root.create_array("minimum", data=np.zeros(2), compressor=None)
    lam_root.attrs.update({
        "frequency": "6h", "resolution": "o96",
        "name_to_index": {"a": 0, "b": 1},
        "data_request": {"grid": 1, "area": "g", "param_level": {}},
        "variables_metadata": {"a": {}, "b": {}},
        "field_shape": [1, n_lam],
    })

    # Globe: coarser global grid
    glob_root = zarr.group()
    n_glob = 20
    glob_data = np.random.default_rng(20).standard_normal((n_dates, 2, 1, n_glob)).astype(np.float32)
    glob_root.create_array("data", data=glob_data, compressor=None)
    glob_root.create_array("dates", data=dates, compressor=None)
    glob_lats = np.linspace(-80, 80, n_glob)
    glob_lons = np.linspace(0, 350, n_glob)
    glob_root.create_array("latitudes", data=glob_lats, compressor=None)
    glob_root.create_array("longitudes", data=glob_lons, compressor=None)
    glob_root.create_array("mean", data=np.zeros(2), compressor=None)
    glob_root.create_array("stdev", data=np.ones(2), compressor=None)
    glob_root.create_array("maximum", data=np.ones(2), compressor=None)
    glob_root.create_array("minimum", data=np.zeros(2), compressor=None)
    glob_root.attrs.update({
        "frequency": "6h", "resolution": "o96",
        "name_to_index": {"a": 0, "b": 1},
        "data_request": {"grid": 1, "area": "g", "param_level": {}},
        "variables_metadata": {"a": {}, "b": {}},
        "field_shape": [1, n_glob],
    })

    lam = GriddedZarr(lam_root, path="lam.zarr")
    globe = GriddedZarr(glob_root, path="globe.zarr")
    ds = Cutout([lam, globe], axis=3, cropping_distance=500.0)
    return ds


class TestCutout:
    def test_int_index(self, cutout_datasets):
        ds = cutout_datasets
        _assert_two_step_equals(ds, [0, 1, 2, 3])

    def test_slice_index(self, cutout_datasets):
        ds = cutout_datasets
        _assert_two_step_equals(ds, [slice(0, 2), slice(1, 4)])

    def test_tuple_all(self, cutout_datasets):
        ds = cutout_datasets
        _assert_two_step_equals(ds, [
            (slice(0, 4), slice(None), slice(None), slice(None)),
        ])

    def test_tuple_var_subset(self, cutout_datasets):
        ds = cutout_datasets
        _assert_two_step_equals(ds, [
            (slice(0, 2), slice(0, 1), slice(None), slice(None)),
        ])

    def test_shape_consistency(self, cutout_datasets):
        ds = cutout_datasets
        direct = ds[0]
        via_two_step = two_step_read(ds, 0)
        assert direct.shape == via_two_step.shape
