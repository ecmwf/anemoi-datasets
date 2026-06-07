# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the two-step read pipeline (ADR-3).

All tests use in-memory zarr groups — no file I/O, no S3, no mocking of zarr.open.
"""

import datetime

import numpy as np
import pytest
import zarr
from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.usage.gridded.select import Select
from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.gridded.subset import Subset
from anemoi.datasets.usage.read_parts import ReadBuffer
from anemoi.datasets.usage.read_parts import ReadPart
from anemoi.datasets.usage.read_parts import execute_parts
from anemoi.datasets.usage.read_parts import factorize
from anemoi.datasets.usage.read_parts import two_step_read

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_DATES = 10
N_VARS = 4
N_ENS = 1
N_GRID = 6


def _make_zarr_group(
    n_dates: int = N_DATES,
    n_vars: int = N_VARS,
    n_ens: int = N_ENS,
    n_grid: int = N_GRID,
    vars: str = "abcd",
    seed: int = 0,
) -> zarr.Group:
    """Create a minimal in-memory zarr group for testing."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_dates, n_vars, n_ens, n_grid)).astype(np.float32)

    root = zarr.group()
    root.create_array("data", data=data, compressor=None)

    start = datetime.datetime(2021, 1, 1)
    freq = datetime.timedelta(hours=6)
    dates = np.array([start + i * freq for i in range(n_dates)], dtype="datetime64")
    root.create_array("dates", data=dates, compressor=None)
    root.create_array("latitudes", data=np.arange(n_grid, dtype=float), compressor=None)
    root.create_array("longitudes", data=np.arange(n_grid, dtype=float), compressor=None)

    # Statistics (required by ZarrStore.statistics)
    root.create_array("mean", data=np.mean(data, axis=(0, 2, 3)), compressor=None)
    root.create_array("stdev", data=np.std(data, axis=(0, 2, 3)), compressor=None)
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


# ---------------------------------------------------------------------------
# ReadPart
# ---------------------------------------------------------------------------


def test_read_part_equality_and_hash():
    arr = np.zeros((5, 3))
    p1 = ReadPart("x", arr, ((0, 2, 1), (0, 3, 1)), ())
    p2 = ReadPart("x", arr, ((0, 2, 1), (0, 3, 1)), ())
    p3 = ReadPart("x", arr, ((1, 3, 1), (0, 3, 1)), ())

    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert p1 != p3
    # Can be used as dict key
    d = {p1: "val"}
    assert d[p2] == "val"


def test_read_part_from_raw_slices():
    arr = np.arange(20).reshape(5, 4)
    slices = (slice(1, 3, 1), slice(0, 4, 1))
    squeeze = (0,)
    p = ReadPart.from_raw_slices("test", arr, slices, squeeze)
    assert p.slices == ((1, 3, 1), (0, 4, 1))
    assert p.squeeze == (0,)


def test_read_part_execute():
    arr = np.arange(20).reshape(5, 4)
    p = ReadPart.from_raw_slices("test", arr, (slice(1, 3, 1), slice(0, 4, 1)), ())
    result = p.execute()
    np.testing.assert_array_equal(result, arr[1:3, 0:4])


def test_read_part_different_array_not_equal():
    a1 = np.zeros((5, 3))
    a2 = np.zeros((5, 3))
    slices = ((0, 5, 1), (0, 3, 1))
    p1 = ReadPart("x", a1, slices, ())
    p2 = ReadPart("x", a2, slices, ())
    # Same data but different array objects → different parts
    assert p1 != p2


# ---------------------------------------------------------------------------
# factorize
# ---------------------------------------------------------------------------


def test_factorize_dedup():
    arr = np.zeros((10, 4, 1, 6))
    slices = ((2, 3, 1), (0, 4, 1), (0, 1, 1), (0, 6, 1))
    p = ReadPart("test", arr, slices, ())

    merged, mapping = factorize([p, p, p])
    assert len(merged) == 1
    assert p in mapping


def test_factorize_merge_adjacent_dates():
    arr = np.zeros((10, 4, 1, 6))
    rest = ((0, 4, 1), (0, 1, 1), (0, 6, 1))
    p0 = ReadPart("t", arr, ((0, 1, 1),) + rest, ())
    p1 = ReadPart("t", arr, ((1, 2, 1),) + rest, ())
    p2 = ReadPart("t", arr, ((2, 3, 1),) + rest, ())

    merged, mapping = factorize([p0, p1, p2])
    assert len(merged) == 1
    mp = merged[0]
    assert mp.slices[0] == (0, 3, 1), f"expected merged date slice (0,3,1), got {mp.slices[0]}"

    assert mapping[p0][1] == 0
    assert mapping[p1][1] == 1
    assert mapping[p2][1] == 2


def test_factorize_no_merge_different_vars():
    arr = np.zeros((10, 4, 1, 6))
    p0 = ReadPart("t", arr, ((0, 1, 1), (0, 2, 1), (0, 1, 1), (0, 6, 1)), ())
    p1 = ReadPart("t", arr, ((1, 2, 1), (2, 4, 1), (0, 1, 1), (0, 6, 1)), ())  # different var slice

    merged, _ = factorize([p0, p1])
    assert len(merged) == 2, "different var slices must NOT be merged"


def test_factorize_no_merge_different_arrays():
    a1 = np.zeros((10, 4))
    a2 = np.zeros((10, 4))
    rest = ((0, 4, 1),)
    p0 = ReadPart("x", a1, ((0, 2, 1),) + rest, ())
    p1 = ReadPart("y", a2, ((0, 2, 1),) + rest, ())

    merged, _ = factorize([p0, p1])
    assert len(merged) == 2, "different zarr arrays must NOT be merged"


# ---------------------------------------------------------------------------
# ReadBuffer
# ---------------------------------------------------------------------------


def test_read_buffer_direct_lookup():
    arr = np.arange(30).reshape(5, 6)
    p = ReadPart("t", arr, ((0, 5, 1), (0, 6, 1)), ())
    raw = {p: arr}
    buffer = ReadBuffer(raw=raw, mapping={})  # identity mapping (no merging)
    result = buffer[p]
    np.testing.assert_array_equal(result, arr)


def test_read_buffer_merged_offset():
    arr = np.arange(60).reshape(10, 6)
    rest = ((0, 6, 1),)

    p0 = ReadPart("t", arr, ((0, 3, 1),) + rest, ())
    p1 = ReadPart("t", arr, ((3, 7, 1),) + rest, ())
    p_merged = ReadPart("t", arr, ((0, 7, 1),) + rest, ())

    raw = {p_merged: arr[0:7]}
    mapping = {p0: (p_merged, 0, None), p1: (p_merged, 3, None)}
    buffer = ReadBuffer(raw=raw, mapping=mapping)

    np.testing.assert_array_equal(buffer[p0], arr[0:3])
    np.testing.assert_array_equal(buffer[p1], arr[3:7])


def test_read_buffer_grid_cols():
    # Merged read holds the union of grid columns; grid_cols selects each part's.
    arr = np.arange(40).reshape(4, 10)
    p_a = ReadPart("t", arr, ((0, 4, 1), (0, 10, 1)), (), grid_index=(1, 3, 5))
    p_b = ReadPart("t", arr, ((0, 4, 1), (0, 10, 1)), (), grid_index=(3, 7))
    union = (1, 3, 5, 7)
    p_merged = ReadPart("t", arr, ((0, 4, 1), (0, 10, 1)), (), grid_index=union)

    raw = {p_merged: arr[:, union]}  # shape (4, 4)
    mapping = {p_a: (p_merged, 0, [0, 1, 2]), p_b: (p_merged, 0, [1, 3])}
    buffer = ReadBuffer(raw=raw, mapping=mapping)

    np.testing.assert_array_equal(buffer[p_a], arr[:, [1, 3, 5]])
    np.testing.assert_array_equal(buffer[p_b], arr[:, [3, 7]])


# ---------------------------------------------------------------------------
# ZarrStore (GriddedZarr leaf node)
# ---------------------------------------------------------------------------


def test_zarr_store_collect_int():
    ds = _make_store()
    parts = ds.collect_read_parts(3)
    assert len(parts) == 1
    p = parts[0]
    assert p.data is ds.data
    # int index → date slice covers exactly one row, squeezed
    assert p.slices[0] == (3, 4, 1)
    assert 0 in p.squeeze  # date axis was an int


def test_zarr_store_collect_slice():
    ds = _make_store()
    parts = ds.collect_read_parts(slice(2, 5))
    assert len(parts) == 1
    p = parts[0]
    assert p.slices[0] == (2, 5, 1)
    assert p.squeeze == ()  # no squeeze for slice


def test_zarr_store_collect_tuple():
    ds = _make_store()
    parts = ds.collect_read_parts((slice(0, 3), slice(1, 3), slice(None), slice(None)))
    assert len(parts) == 1
    p = parts[0]
    assert p.slices[0] == (0, 3, 1)
    assert p.slices[1] == (1, 3, 1)


def test_zarr_store_read_from_buffer_matches_getitem():
    ds = _make_store()
    for n in [0, 3, slice(2, 5), (slice(1, 4), slice(0, 2), slice(None), slice(None))]:
        parts = ds.collect_read_parts(n)
        factorized, part_map = factorize(parts)
        raw = execute_parts(factorized)
        buffer = ReadBuffer(raw=raw, mapping=part_map)

        result_new = ds.read_from_buffer(n, buffer)
        result_old = ds[n]
        np.testing.assert_array_equal(result_new, result_old, err_msg=f"mismatch for index {n!r}")


def test_two_step_read_zarr_store():
    ds = _make_store()
    for n in [0, 5, slice(3, 7), (slice(0, 5), slice(1, 3), slice(None), slice(None))]:
        result = two_step_read(ds, n)
        expected = ds[n]
        np.testing.assert_array_equal(result, expected, err_msg=f"two_step_read mismatch for {n!r}")


# ---------------------------------------------------------------------------
# Select wrapper
# ---------------------------------------------------------------------------


def _make_select(var_indices=(0, 2), **kwargs) -> Select:
    ds = _make_store(**kwargs)
    return Select(ds, list(var_indices), reason={"select": var_indices})


def test_select_collect_int():
    sel = _make_select(var_indices=[0, 2])
    parts = sel.collect_read_parts(3)
    assert len(parts) == 1
    # Variable slice should be all vars (0:4), NOT just [0,2]
    p = parts[0]
    assert p.slices[1] == (0, N_VARS, 1), f"expected all vars, got {p.slices[1]}"


def test_select_collect_tuple_expands_vars():
    sel = _make_select(var_indices=[1, 3])
    # Request only first 2 vars in outer space — inner must request all
    n = (slice(0, 3), slice(0, 2), slice(None), slice(None))
    parts = sel.collect_read_parts(n)
    assert len(parts) == 1
    assert parts[0].slices[1] == (0, N_VARS, 1), "inner read must request all vars"


def test_select_read_from_buffer_matches_getitem():
    sel = _make_select(var_indices=[0, 2])
    for n in [0, slice(2, 5), (slice(0, 4), slice(0, 2), slice(None), slice(None))]:
        parts = sel.collect_read_parts(n)
        factorized, part_map = factorize(parts)
        raw = execute_parts(factorized)
        buffer = ReadBuffer(raw=raw, mapping=part_map)

        result_new = sel.read_from_buffer(n, buffer)
        result_old = sel[n]
        np.testing.assert_array_equal(result_new, result_old, err_msg=f"Select mismatch for {n!r}")


def test_two_step_read_select():
    sel = _make_select(var_indices=[1, 3])
    for n in [0, 4, slice(1, 6), (slice(0, 5), slice(None), slice(None), slice(None))]:
        result = two_step_read(sel, n)
        expected = sel[n]
        np.testing.assert_array_equal(result, expected, err_msg=f"Select two_step_read mismatch for {n!r}")


# ---------------------------------------------------------------------------
# Subset wrapper
# ---------------------------------------------------------------------------


def _make_subset(indices=(0, 2, 4, 6, 8), **kwargs) -> Subset:
    ds = _make_store(**kwargs)
    return Subset(ds, list(indices), reason={"start": None, "end": None})


def test_subset_collect_int():
    sub = _make_subset(indices=[0, 2, 4, 6, 8])
    parts = sub.collect_read_parts(2)  # outer 2 → inner 4
    assert len(parts) == 1
    p = parts[0]
    assert p.slices[0][0] == 4, f"expected inner date index 4, got {p.slices[0][0]}"


def test_subset_collect_contiguous_maps_to_slice():
    # indices [0,1,2,3,4] are contiguous → inner should be a slice
    sub = _make_subset(indices=[0, 1, 2, 3, 4])
    parts = sub.collect_read_parts(slice(0, 3))
    assert len(parts) == 1
    # inner indices [0,1,2] → slice(0,3) → single part
    assert parts[0].slices[0] == (0, 3, 1)


def test_subset_collect_non_contiguous_multi_parts():
    # indices [0,3,7] are non-contiguous → inner is a list → 3 separate parts
    sub = _make_subset(indices=[0, 3, 7])
    parts = sub.collect_read_parts(slice(0, 3))  # all 3 outer indices
    # non-contiguous → cannot be merged into one slice, collect_read_parts returns 3 parts
    assert len(parts) == 3


def test_subset_read_from_buffer_matches_getitem():
    sub = _make_subset(indices=[0, 2, 4, 6, 8])
    for n in [0, 2, slice(1, 4), (slice(0, 3), slice(None), slice(None), slice(None))]:
        parts = sub.collect_read_parts(n)
        factorized, part_map = factorize(parts)
        raw = execute_parts(factorized)
        buffer = ReadBuffer(raw=raw, mapping=part_map)

        result_new = sub.read_from_buffer(n, buffer)
        result_old = sub[n]
        np.testing.assert_array_equal(result_new, result_old, err_msg=f"Subset mismatch for {n!r}")


def test_two_step_read_subset():
    sub = _make_subset(indices=[0, 2, 4, 6, 8])
    for n in [0, 3, slice(0, 4), (slice(1, 4), slice(None), slice(None), slice(None))]:
        result = two_step_read(sub, n)
        expected = sub[n]
        np.testing.assert_array_equal(result, expected, err_msg=f"Subset two_step_read mismatch for {n!r}")


# ---------------------------------------------------------------------------
# Select + Subset + ZarrStore chain
# ---------------------------------------------------------------------------


def _make_select_subset_chain() -> Select:
    """Subset(ZarrStore) → Select(Subset(ZarrStore))."""
    ds = _make_store()
    sub = Subset(ds, list(range(0, N_DATES, 2)), reason={})  # every other date
    sel = Select(sub, [0, 2], reason={"select": [0, 2]})  # 2 of 4 vars
    return sel


def test_chain_two_step_read_int():
    chain = _make_select_subset_chain()
    for i in range(len(chain)):
        result = two_step_read(chain, i)
        expected = chain[i]
        np.testing.assert_array_equal(result, expected, err_msg=f"chain mismatch at i={i}")


def test_chain_two_step_read_slice():
    chain = _make_select_subset_chain()
    for s in [slice(0, 3), slice(1, 4, 2), slice(None)]:
        result = two_step_read(chain, s)
        expected = chain[s]
        np.testing.assert_array_equal(result, expected, err_msg=f"chain mismatch for slice {s!r}")


def test_chain_two_step_read_tuple():
    chain = _make_select_subset_chain()
    indices = [
        (slice(0, 3), slice(None), slice(None), slice(None)),
        (slice(1, 4), slice(0, 1), slice(None), slice(None)),
        (0, slice(None), 0, slice(None)),
    ]
    for n in indices:
        result = two_step_read(chain, n)
        expected = chain[n]
        np.testing.assert_array_equal(result, expected, err_msg=f"chain tuple mismatch for {n!r}")


# ---------------------------------------------------------------------------
# Factorization of batch reads (simulate training loop)
# ---------------------------------------------------------------------------


def test_batch_factorized_to_single_read():
    """Simulate reading a batch [ds[i], ds[i+1], ds[i+2]] — should merge to one zarr read."""
    ds = _make_store()

    all_parts = []
    for i in range(3, 6):
        all_parts.extend(ds.collect_read_parts(i))

    assert len(all_parts) == 3, "one part per date"
    merged, mapping = factorize(all_parts)
    assert len(merged) == 1, "three adjacent date reads should merge to one"
    mp = merged[0]
    assert mp.slices[0] == (3, 6, 1), f"merged date range should be (3,6,1), got {mp.slices[0]}"

    raw = execute_parts(merged)
    buffer = ReadBuffer(raw=raw, mapping=mapping)

    for i, outer_i in enumerate(range(3, 6)):
        result = ds.read_from_buffer(outer_i, buffer)
        expected = ds[outer_i]
        np.testing.assert_array_equal(result, expected, err_msg=f"batch mismatch at i={outer_i}")


def test_non_adjacent_batch_not_over_merged():
    """Non-contiguous reads: bounding-box merge still works but reads extra rows."""
    ds = _make_store()

    parts = []
    for i in [0, 5, 9]:
        parts.extend(ds.collect_read_parts(i))

    merged, mapping = factorize(parts)
    assert len(merged) == 1, "all from same zarr array → single merged part"
    mp = merged[0]
    # Bounding box: 0..10
    assert mp.slices[0] == (0, 10, 1), f"expected (0,10,1) bounding box, got {mp.slices[0]}"

    raw = execute_parts(merged)
    buffer = ReadBuffer(raw=raw, mapping=mapping)

    for i in [0, 5, 9]:
        result = ds.read_from_buffer(i, buffer)
        expected = ds[i]
        np.testing.assert_array_equal(result, expected, err_msg=f"non-adj batch mismatch at i={i}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
