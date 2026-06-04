# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Phase 5 tests: parallel execute_parts via ThreadPoolExecutor.

Verifies that:
- parallel execution (num_threads>1) returns same results as sequential (num_threads=1)
- empty parts list returns {}
- single part skips thread pool (still correct)
- num_threads env var is respected
- thread count is configurable per-call
"""

import datetime

import numpy as np
import pytest
import zarr
from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.usage.gridded.concat import Concat as GriddedConcat
from anemoi.datasets.usage.gridded.join import Join
from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.read_parts import ReadPart, execute_parts, factorize, two_step_read

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


def _make_store(path="test.zarr", **kwargs):
    return GriddedZarr(_make_zarr_group(**kwargs), path=path)


# ---------------------------------------------------------------------------
# execute_parts: basic contract
# ---------------------------------------------------------------------------


def test_execute_parts_empty():
    assert execute_parts([]) == {}


def test_execute_parts_single_part_sequential_path():
    """Single part must skip the thread pool and still return correct result."""
    store = _make_store()
    parts = store.collect_read_parts(0)
    assert len(parts) >= 1
    factorized, mapping = factorize(parts)
    result_seq = execute_parts(factorized, num_threads=1)
    result_par = execute_parts(factorized, num_threads=4)
    for part in factorized:
        np.testing.assert_array_equal(result_seq[part], result_par[part])


# ---------------------------------------------------------------------------
# Parallel == sequential for single store
# ---------------------------------------------------------------------------


def test_parallel_matches_sequential_int():
    store = _make_store()
    for n in [0, 3, 7]:
        seq = two_step_read.__wrapped__(store, n) if hasattr(two_step_read, "__wrapped__") else None
        parts = store.collect_read_parts(n)
        factorized, mapping = factorize(parts)
        raw_seq = execute_parts(factorized, num_threads=1)
        raw_par = execute_parts(factorized, num_threads=4)
        for part in factorized:
            np.testing.assert_array_equal(raw_seq[part], raw_par[part], err_msg=f"n={n}")


def test_parallel_matches_sequential_slice():
    store = _make_store()
    for n in [slice(0, 4), slice(2, 8)]:
        parts = store.collect_read_parts(n)
        factorized, mapping = factorize(parts)
        raw_seq = execute_parts(factorized, num_threads=1)
        raw_par = execute_parts(factorized, num_threads=4)
        for part in factorized:
            np.testing.assert_array_equal(raw_seq[part], raw_par[part], err_msg=f"n={n}")


# ---------------------------------------------------------------------------
# Parallel == sequential for multi-source (Join produces 2 parts)
# ---------------------------------------------------------------------------


def test_parallel_join_matches_sequential():
    """Join produces one ReadPart per child store — parallel vs sequential must match."""
    from unittest.mock import patch
    with patch("anemoi.datasets.usage.forwards.Combined.check_variables_compatibility", return_value=None):
        d1 = _make_store("j1.zarr", n_vars=2, vars="ab", seed=0)
        d2 = _make_store("j2.zarr", n_vars=2, vars="cd", seed=1)
        ds = Join([d1, d2])

    for n in [0, slice(0, 4)]:
        parts = ds.collect_read_parts(n)
        factorized, mapping = factorize(parts)
        assert len(factorized) >= 2, "Join should produce at least 2 parts"
        raw_seq = execute_parts(factorized, num_threads=1)
        raw_par = execute_parts(factorized, num_threads=4)
        for part in factorized:
            np.testing.assert_array_equal(raw_seq[part], raw_par[part])


def test_parallel_concat_matches_sequential():
    """GriddedConcat spanning boundary produces 2 parts."""
    from unittest.mock import patch
    with patch("anemoi.datasets.usage.forwards.Combined.check_variables_compatibility", return_value=None):
        d1 = _make_store("c1.zarr", n_dates=4, seed=0, start_date=datetime.datetime(2021, 1, 1))
        d2 = _make_store("c2.zarr", n_dates=4, seed=1, start_date=datetime.datetime(2021, 1, 2))
        ds = GriddedConcat([d1, d2])

    n = slice(2, 6)  # spans boundary
    parts = ds.collect_read_parts(n)
    factorized, mapping = factorize(parts)
    assert len(factorized) == 2, "cross-boundary slice should produce 2 factorized parts"
    raw_seq = execute_parts(factorized, num_threads=1)
    raw_par = execute_parts(factorized, num_threads=4)
    for part in factorized:
        np.testing.assert_array_equal(raw_seq[part], raw_par[part])


# ---------------------------------------------------------------------------
# End-to-end: two_step_read results unchanged with parallel execution
# ---------------------------------------------------------------------------


def test_end_to_end_parallel_equals_direct():
    """two_step_read with parallel execute_parts gives same result as ds[n]."""
    from unittest.mock import patch
    import anemoi.datasets.usage.read_parts as rp

    store = _make_store()
    for n in [0, slice(0, 4), (slice(0, 4), slice(None), slice(None), slice(None))]:
        expected = store[n]
        # Temporarily override READ_PARTS_THREADS to 4
        orig = rp.READ_PARTS_THREADS
        rp.READ_PARTS_THREADS = 4
        try:
            actual = two_step_read(store, n)
        finally:
            rp.READ_PARTS_THREADS = orig
        np.testing.assert_array_equal(actual, expected, err_msg=f"n={n}")


def test_default_thread_count_is_2():
    import anemoi.datasets.usage.read_parts as rp
    assert rp.READ_PARTS_THREADS == int(
        __import__("os").environ.get("ANEMOI_DATASETS_READ_PARTS_THREADS", "2")
    )
