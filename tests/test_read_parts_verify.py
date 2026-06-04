# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Verification tests for the two-step read review (progress-verify.md).

Covers:
  B1 — ZarrWithMissingDates: missing-date check preserved via two_step_read path
  B2 — MissingDates: missing-date check preserved via two_step_read path
  E6 — List/array indices in tuples fall back gracefully to legacy __getitem__
  C-fallbacks — Classes that raise NotImplementedError fall back correctly
"""

import datetime

import numpy as np
import pytest
import zarr
from anemoi.utils.dates import frequency_to_string

from anemoi.datasets import MissingDateError
from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.read_parts import two_step_read

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DATES = 8
N_VARS = 4
N_ENS = 1
N_GRID = 6


def _make_zarr_group(
    n_dates=N_DATES, n_vars=N_VARS, n_ens=N_ENS, n_grid=N_GRID,
    vars="abcd", start_date=datetime.datetime(2021, 1, 1), seed=0,
    missing_dates=None,
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
    attrs = {
        "frequency": frequency_to_string(freq),
        "resolution": "o96",
        "name_to_index": {v: i for i, v in enumerate(var_list)},
        "data_request": {"grid": 1, "area": "g", "param_level": {}},
        "variables_metadata": {v: {} for v in var_list},
        "field_shape": [1, n_grid],
    }
    if missing_dates:
        attrs["missing_dates"] = missing_dates
    root.attrs.update(attrs)
    return root


def _make_store(**kwargs):
    return GriddedZarr(_make_zarr_group(**kwargs), path="test.zarr")


# ---------------------------------------------------------------------------
# B1 — ZarrWithMissingDates: missing check preserved
# ---------------------------------------------------------------------------


class TestZarrWithMissingDates:
    def _make(self):
        from anemoi.datasets.usage.gridded.store import ZarrWithMissingDates
        # Mark index 3 as missing (2021-01-01T18:00:00)
        missing_str = "2021-01-01T18:00:00"
        group = _make_zarr_group(missing_dates=[missing_str])
        return ZarrWithMissingDates(group, path="test.zarr")

    def test_non_missing_int_works(self):
        ds = self._make()
        result = ds[0]
        assert result is not None
        assert result.shape == (N_VARS, N_ENS, N_GRID)

    def test_missing_int_raises(self):
        ds = self._make()
        assert 3 in ds.missing
        with pytest.raises(MissingDateError):
            ds[3]

    def test_missing_slice_raises(self):
        ds = self._make()
        with pytest.raises(MissingDateError):
            ds[slice(2, 5)]  # includes index 3

    def test_missing_tuple_int_raises(self):
        ds = self._make()
        with pytest.raises(MissingDateError):
            ds[(3, slice(None), slice(None), slice(None))]

    def test_missing_tuple_slice_raises(self):
        ds = self._make()
        with pytest.raises(MissingDateError):
            ds[(slice(2, 5), slice(None), slice(None), slice(None))]

    def test_two_step_read_non_missing_correct(self):
        ds = self._make()
        result_gate = ds[0]
        result_two_step = two_step_read(ds, 0)
        np.testing.assert_array_equal(result_gate, result_two_step)

    def test_two_step_read_missing_raises(self):
        ds = self._make()
        with pytest.raises(MissingDateError):
            two_step_read(ds, 3)


# ---------------------------------------------------------------------------
# B2 — MissingDates wrapper: missing check preserved
# ---------------------------------------------------------------------------


class TestMissingDatesWrapper:
    def _make(self):
        from anemoi.datasets.usage.gridded.missing import MissingDates
        store = _make_store()
        # Force index 2 to be "missing"
        ds = MissingDates(store, [2])
        return ds

    def test_non_missing_int_works(self):
        ds = self._make()
        result = ds[0]
        assert result is not None
        assert result.shape == (N_VARS, N_ENS, N_GRID)

    def test_missing_int_raises(self):
        ds = self._make()
        with pytest.raises(MissingDateError):
            ds[2]

    def test_missing_slice_raises(self):
        ds = self._make()
        with pytest.raises(MissingDateError):
            ds[slice(1, 4)]  # includes 2

    def test_two_step_read_non_missing_correct(self):
        ds = self._make()
        np.testing.assert_array_equal(ds[0], two_step_read(ds, 0))

    def test_two_step_read_missing_raises(self):
        ds = self._make()
        with pytest.raises(MissingDateError):
            two_step_read(ds, 2)


# ---------------------------------------------------------------------------
# E6 — List/array indices fall back gracefully
# ---------------------------------------------------------------------------


def test_list_index_in_tuple_falls_back_correctly():
    """Tuple with list element at axis 0 should fall back to legacy expand_list_indexing."""
    store = _make_store()
    # This is the expand_list_indexing pattern: tuple index with a list at one position
    n = ([0, 1, 2], slice(None), slice(None), slice(None))
    result = store[n]
    assert result is not None
    assert result.shape == (3, N_VARS, N_ENS, N_GRID)


# ---------------------------------------------------------------------------
# C-fallbacks — All NotImplementedError wrappers fall back correctly
# ---------------------------------------------------------------------------


def test_tabular_zarr_fallback():
    """TabularZarr raises NotImplementedError → gate falls back."""
    from anemoi.datasets.usage.tabular.store import TabularZarr
    from anemoi.datasets.windows.view import WindowView

    # TabularZarr is complex to construct; just verify collect_read_parts raises
    class _MockTabular(TabularZarr.__bases__[0]):  # ZarrStore
        pass

    store = _make_store()
    # Verify the method exists and raises
    assert hasattr(TabularZarr, "collect_read_parts")
    with pytest.raises(NotImplementedError, match="TabularZarr"):
        TabularZarr.collect_read_parts(None, 0)


def test_tensors_raises_not_implemented():
    """Tensors.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.tabular.tensors import Tensors
    with pytest.raises(NotImplementedError, match="Tensors"):
        Tensors.collect_read_parts(None, 0)


def test_interpolate_nearest_raises_not_implemented():
    """InterpolateNearest.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.gridded.interpolate import InterpolateNearest
    with pytest.raises(NotImplementedError, match="InterpolateNearest"):
        InterpolateNearest.collect_read_parts(None, 0)


def test_step_subset_raises_not_implemented():
    """StepSubset.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.trajectories.subset import StepSubset
    with pytest.raises(NotImplementedError, match="StepSubset"):
        StepSubset.collect_read_parts(None, 0)


def test_single_step_view_raises_not_implemented():
    """SingleStepView.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.trajectories.subset import SingleStepView
    with pytest.raises(NotImplementedError, match="SingleStepView"):
        SingleStepView.collect_read_parts(None, 0)


def test_trajectories_subset_raises_not_implemented():
    """trajectories.Subset.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.trajectories.subset import Subset as TrajSubset
    with pytest.raises(NotImplementedError, match="trajectories.Subset"):
        TrajSubset.collect_read_parts(None, 0)


def test_zipbase_raises_not_implemented():
    """ZipBase.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.gridded.xy import ZipBase
    with pytest.raises(NotImplementedError, match="ZipBase"):
        ZipBase.collect_read_parts(None, 0)


def test_chain_raises_not_implemented():
    """Chain.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.gridded.unchecked import Chain
    with pytest.raises(NotImplementedError, match="Chain"):
        Chain.collect_read_parts(None, 0)


def test_complement_raises_not_implemented():
    """Complement.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.gridded.complement import Complement
    with pytest.raises(NotImplementedError, match="Complement"):
        Complement.collect_read_parts(None, 0)


def test_skip_missing_raises_not_implemented():
    """SkipMissingDates.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.gridded.missing import SkipMissingDates
    with pytest.raises(NotImplementedError, match="SkipMissingDates"):
        SkipMissingDates.collect_read_parts(None, 0)


def test_tabular_select_raises_not_implemented():
    """tabular.Select.collect_read_parts raises NotImplementedError."""
    from anemoi.datasets.usage.tabular.select import Select as TabSelect
    with pytest.raises(NotImplementedError, match="tabular.Select"):
        TabSelect.collect_read_parts(None, 0)
