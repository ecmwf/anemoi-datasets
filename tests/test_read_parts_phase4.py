# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Phase 4 tests: always-on gate, new wrappers, fallback for complex wrappers.

Phase 4 changes:
- Gate is unconditional: ds[n] always tries two_step_read first
- collect_read_parts returns None → fall back to eager __getitem__
- Rescale, Masked, Ensemble (Number) now have collect_read_parts + read_from_buffer
- RollingAverage, FillMissing, InterpolateFrequency return None → fallback
"""

import datetime
from unittest.mock import patch

import numpy as np
import pytest
import zarr
from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.read_parts import two_step_read

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DATES = 8
N_VARS = 4
N_ENS = 2
N_GRID = 6


def _make_zarr_group(
    n_dates=N_DATES, n_vars=N_VARS, n_ens=N_ENS, n_grid=N_GRID,
    vars="abcd", start_date=datetime.datetime(2021, 1, 1), seed=0,
):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_dates, n_vars, n_ens, n_grid)).astype(np.float32)
    root = zarr.group()
    root.create_array("data", data=data, compressor=None)
    freq = datetime.timedelta(hours=6)
    dates = np.array([start_date + i * freq for i in range(n_dates)], dtype="datetime64")
    root.create_array("dates", data=dates, compressor=None)
    root.create_array("latitudes", data=np.linspace(-90, 90, n_grid), compressor=None)
    root.create_array("longitudes", data=np.linspace(0, 360, n_grid, endpoint=False), compressor=None)
    root.create_array("mean", data=np.zeros(n_vars), compressor=None)
    root.create_array("stdev", data=np.ones(n_vars), compressor=None)
    root.create_array("maximum", data=np.ones(n_vars), compressor=None)
    root.create_array("minimum", data=np.zeros(n_vars), compressor=None)
    var_list = list(vars[:n_vars])
    root.attrs.update({
        "frequency": frequency_to_string(freq),
        "resolution": "o96",
        "name_to_index": {v: i for i, v in enumerate(var_list)},
        "data_request": {"grid": 1, "area": "g", "param_level": {}},
        "variables_metadata": {v: {} for v in var_list},
        "field_shape": [1, n_grid],
    })
    return root


def _make_store(**kwargs):
    return GriddedZarr(_make_zarr_group(**kwargs), path="test.zarr")


# ---------------------------------------------------------------------------
# Gate is always on: ds[n] == two_step_read(ds, n)
# ---------------------------------------------------------------------------


def test_gate_always_on():
    """ds[n] and two_step_read(ds, n) return identical results (gate is unconditional)."""
    ds = _make_store()
    for n in [0, 3, slice(1, 5), (slice(0, 4), slice(1, 3), slice(None), slice(None))]:
        np.testing.assert_array_equal(ds[n], two_step_read(ds, n))


def test_gate_fallback_when_collect_returns_none():
    """Gate falls back to eager __getitem__ when collect_read_parts returns None."""
    ds = _make_store()
    original = ds.collect_read_parts

    def _unsupported(n):
        return None

    ds.collect_read_parts = _unsupported
    try:
        result = ds[3]
        assert result is not None
        assert result.shape == (N_VARS, N_ENS, N_GRID)
    finally:
        ds.collect_read_parts = original


# ---------------------------------------------------------------------------
# Rescale
# ---------------------------------------------------------------------------


class TestRescale:
    def _ds_and_store(self):
        store = _make_store()
        from anemoi.datasets.usage.gridded.rescale import Rescale
        # Rescale var 'a' (index 0) by scale=2, offset=1
        ds = Rescale(store, {"a": (2.0, 1.0)})
        return ds, store

    def test_int_matches_manual(self):
        ds, store = self._ds_and_store()
        raw = store[0]  # (N_VARS, N_ENS, N_GRID)
        expected = raw.copy()
        expected[0] = raw[0] * 2.0 + 1.0
        actual = two_step_read(ds, 0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_slice(self):
        ds, _ = self._ds_and_store()
        np.testing.assert_array_equal(ds[slice(0, 4)], two_step_read(ds, slice(0, 4)))

    def test_tuple(self):
        ds, _ = self._ds_and_store()
        n = (slice(0, 4), slice(None), slice(None), slice(None))
        np.testing.assert_array_equal(ds[n], two_step_read(ds, n))

    def test_tuple_var_subset(self):
        ds, _ = self._ds_and_store()
        n = (slice(0, 4), slice(0, 2), slice(None), slice(None))
        np.testing.assert_array_equal(ds[n], two_step_read(ds, n))

    def test_gate_transparent(self):
        """ds[n] is same as two_step_read since gate is on."""
        ds, _ = self._ds_and_store()
        for n in [0, slice(0, 3)]:
            np.testing.assert_array_equal(ds[n], two_step_read(ds, n))


# ---------------------------------------------------------------------------
# Masked
# ---------------------------------------------------------------------------


class TestMasked:
    def _ds_and_store(self):
        store = _make_store()
        from anemoi.datasets.usage.gridded.masked import Masked
        from anemoi.datasets.usage.debug import Node

        class _ConcreteMasked(Masked):
            def tree(self):
                return Node(self, [self.forward.tree()])
            def forwards_subclass_metadata_specific(self):
                return {}

        mask = np.array([True, False, True, False, True, False])  # 3 of 6 grid points
        ds = _ConcreteMasked(store, mask)
        return ds, store, mask

    def test_int_matches_manual(self):
        ds, store, mask = self._ds_and_store()
        raw = store[0]  # (N_VARS, N_ENS, N_GRID)
        expected = raw[..., mask]
        actual = two_step_read(ds, 0)
        np.testing.assert_array_equal(actual, expected)

    def test_slice(self):
        ds, _, _ = self._ds_and_store()
        np.testing.assert_array_equal(ds[slice(0, 4)], two_step_read(ds, slice(0, 4)))

    def test_tuple(self):
        ds, _, _ = self._ds_and_store()
        n = (slice(0, 4), slice(None), slice(None), slice(None))
        np.testing.assert_array_equal(ds[n], two_step_read(ds, n))

    def test_shape(self):
        ds, _, mask = self._ds_and_store()
        result = two_step_read(ds, 0)
        assert result.shape == (N_VARS, N_ENS, int(np.sum(mask)))

    def test_gate_transparent(self):
        ds, _, _ = self._ds_and_store()
        for n in [0, slice(0, 3)]:
            np.testing.assert_array_equal(ds[n], two_step_read(ds, n))


# ---------------------------------------------------------------------------
# Ensemble (Number)
# ---------------------------------------------------------------------------


class TestEnsemble:
    def _ds_and_store(self):
        store = _make_store()  # N_ENS=2
        from anemoi.datasets.usage.gridded.ensemble import Number
        # Select member index 1 (zero-based) = member number 2 (one-based)
        ds = Number(store, members=[1])
        return ds, store

    def test_int_matches_manual(self):
        ds, store = self._ds_and_store()
        raw = store[0]  # (N_VARS, N_ENS, N_GRID) = (4, 2, 6)
        expected = raw[:, [1], :]  # select member index 1
        actual = two_step_read(ds, 0)
        np.testing.assert_array_equal(actual, expected)

    def test_slice(self):
        ds, _ = self._ds_and_store()
        np.testing.assert_array_equal(ds[slice(0, 4)], two_step_read(ds, slice(0, 4)))

    def test_tuple(self):
        ds, _ = self._ds_and_store()
        n = (slice(0, 4), slice(None), slice(None), slice(None))
        np.testing.assert_array_equal(ds[n], two_step_read(ds, n))

    def test_shape(self):
        ds, _ = self._ds_and_store()
        result = two_step_read(ds, 0)
        assert result.shape == (N_VARS, 1, N_GRID), result.shape


# ---------------------------------------------------------------------------
# Fallback for complex wrappers
# ---------------------------------------------------------------------------


def test_rolling_average_falls_back():
    """RollingAverage returns None from collect_read_parts → gate falls back to eager."""
    from anemoi.datasets.usage.gridded.rolling_average import RollingAverage
    store = _make_store(n_dates=10)
    ds = RollingAverage(store, window=(-1, 1, "freq"))
    # Should not raise — fallback to eager __getitem__
    result = ds[2]
    assert result is not None
    assert result.shape == (N_VARS, N_ENS, N_GRID)


def test_fill_missing_falls_back():
    """MissingDatesFill returns None from collect_read_parts → gate falls back to eager."""
    from anemoi.datasets.usage.gridded.fill_missing import MissingDatesClosest
    store = _make_store(n_dates=10)
    # Mark index 3 as missing
    ds = MissingDatesClosest(store, closest="previous")
    # Access a non-missing date — should fall back to eager without error
    result = ds[0]
    assert result is not None


def test_interpolate_falls_back():
    """InterpolateFrequency returns None from collect_read_parts → gate falls back to eager."""
    from anemoi.datasets.usage.gridded.interpolate import InterpolateFrequency
    store = _make_store(n_dates=8)
    ds = InterpolateFrequency(store, frequency=datetime.timedelta(hours=3))
    result = ds[0]
    assert result is not None
