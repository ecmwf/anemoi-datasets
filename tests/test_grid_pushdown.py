# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Workstream D — grid-subset pushdown.

When a cutout (or a plain store) is indexed with a *subset* of the grid axis,
the two-step read should fetch only the needed grid points (via a ``grid_index``
``ReadPart`` and zarr orthogonal indexing), not the full grid.  Two things are
checked:

1. **Correctness** — pushdown output is byte-identical to the eager full-read
   path (obtained with the ``READ_PARTS_ENABLED`` kill-switch off).
2. **Efficiency** — the collected/factorized parts read exactly the requested
   number of grid points (≪ the full grid).
"""

import datetime
from contextlib import contextmanager

import numpy as np
import pytest
import zarr

import anemoi.datasets.usage.read_parts as rp
from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.read_parts import factorize
from anemoi.datasets.usage.read_parts import two_step_read

N_DATES = 4


@contextmanager
def eager_path():
    """Force the eager recursive __getitem__ (two-step off) inside the block."""
    saved = rp.READ_PARTS_ENABLED
    rp.READ_PARTS_ENABLED = False
    try:
        yield
    finally:
        rp.READ_PARTS_ENABLED = saved


def _group(seed, lats, lons, n_vars=2, grid_chunk=None):
    n = len(lats)
    data = np.random.default_rng(seed).standard_normal((N_DATES, n_vars, 1, n)).astype(np.float32)
    root = zarr.group()
    if grid_chunk is None:
        root.create_array("data", data=data, compressor=None)
    else:
        root.create_dataset("data", data=data, chunks=(1, n_vars, 1, grid_chunk), compressor=None)
    freq = datetime.timedelta(hours=6)
    dates = np.array([datetime.datetime(2021, 1, 1) + i * freq for i in range(N_DATES)], dtype="datetime64")
    root.create_array("dates", data=dates, compressor=None)
    root.create_array("latitudes", data=lats, compressor=None)
    root.create_array("longitudes", data=lons, compressor=None)
    root.create_array("mean", data=np.zeros(n_vars), compressor=None)
    root.create_array("stdev", data=np.ones(n_vars), compressor=None)
    root.create_array("maximum", data=np.ones(n_vars), compressor=None)
    root.create_array("minimum", data=np.zeros(n_vars), compressor=None)
    root.attrs.update({
        "frequency": "6h", "resolution": "o96",
        "name_to_index": {"a": 0, "b": 1},
        "data_request": {"grid": 1, "area": "g", "param_level": {}},
        "variables_metadata": {"a": {}, "b": {}},
        "field_shape": [1, n],
    })
    return root


def _grid_points_read(parts):
    """Total grid points the (factorized) parts touch on the last axis."""
    total = 0
    for p in parts:
        if p.grid_index is not None:
            total += len(p.grid_index)
        else:
            s, e, t = p.slices[-1]
            total += len(range(s, e, t))
    return total


@pytest.fixture()
def cutout():
    pytest.importorskip("anemoi.transform.spatial")
    from anemoi.datasets.usage.gridded.grids import Cutout

    lam = GriddedZarr(
        _group(10, np.array([5.0, 10.0, 15.0, 5.0, 10.0]), np.array([5.0, 5.0, 5.0, 10.0, 10.0])),
        path="lam.zarr",
    )
    globe = GriddedZarr(
        _group(20, np.linspace(-80, 80, 20), np.linspace(0, 350, 20)),
        path="globe.zarr",
    )
    return Cutout([lam, globe], axis=3, cropping_distance=500.0)


class TestChunkedGridGuard:
    """When the grid axis is chunked, pushdown engages even for a shard that
    touches every store (the store-skip guard is for one-chunk grids only)."""

    def _chunked_cutout(self):
        pytest.importorskip("anemoi.transform.spatial")
        from anemoi.datasets.usage.gridded.grids import Cutout

        lam = GriddedZarr(_group(10, np.array([5.0, 10.0, 15.0, 5.0, 10.0]),
                                 np.array([5.0, 5.0, 5.0, 10.0, 10.0]), grid_chunk=2), path="lam.zarr")
        globe = GriddedZarr(_group(20, np.linspace(-80, 80, 20), np.linspace(0, 350, 20),
                                   grid_chunk=4), path="globe.zarr")
        return Cutout([lam, globe], axis=3, cropping_distance=500.0)

    def test_spanning_shard_pushes_down_when_grid_chunked(self):
        ds = self._chunked_cutout()
        assert ds._grid_is_chunked
        total = ds.shape[-1]
        lam_len = ds.lams[0].shape[-1]
        n = (0, slice(None), slice(None), slice(lam_len - 1, lam_len + 2))  # spans both stores
        parts, _ = factorize(ds.collect_read_parts(n))
        assert all(p.grid_index is not None for p in parts), [str(p) for p in parts]
        with eager_path():
            expected = ds[n]
        np.testing.assert_array_equal(two_step_read(ds, n), expected)

    def test_one_chunk_grid_still_guards(self, cutout):
        # default fixture = one grid chunk → spanning still falls back
        assert not cutout._grid_is_chunked


class TestPushdownThroughWrappers:
    """Pushdown must reach the leaf through Select (var subset) and Subset (date
    subset) constituents — the real-cutout case (members opened with select/adjust)."""

    def _wrapped_cutout(self, wrap):
        pytest.importorskip("anemoi.transform.spatial")
        from anemoi.datasets.usage.gridded.grids import Cutout

        lam = wrap(GriddedZarr(_group(10, np.array([5.0, 10.0, 15.0, 5.0, 10.0]),
                                      np.array([5.0, 5.0, 5.0, 10.0, 10.0])), path="lam.zarr"))
        globe = wrap(GriddedZarr(_group(20, np.linspace(-80, 80, 20), np.linspace(0, 350, 20)),
                                 path="globe.zarr"))
        return Cutout([lam, globe], axis=3, cropping_distance=500.0)

    def _check(self, ds):
        assert ds._pushdown_supported, "wrapper not grid-transparent"
        total = ds.shape[-1]
        lam_len = ds.lams[0].shape[-1]
        # LAM-region and globe-region shards both skip a store → push down.
        for sh in [slice(0, 2), slice(lam_len, total)]:
            n = (0, slice(None), slice(None), sh)
            parts, _ = factorize(ds.collect_read_parts(n))
            assert all(p.grid_index is not None for p in parts), [str(p) for p in parts]
            with eager_path():
                expected = ds[n]
            np.testing.assert_array_equal(two_step_read(ds, n), expected)

    def test_select_wrapped(self):
        from anemoi.datasets.usage.gridded.select import Select

        self._check(self._wrapped_cutout(lambda d: Select(d, [0], reason={"select": [0]})))

    def test_subset_wrapped(self):
        from anemoi.datasets.usage.gridded.subset import Subset

        self._check(self._wrapped_cutout(lambda d: Subset(d, [0, 1, 2], reason={})))


class TestCutoutPushdownCorrectness:
    def _indices(self, total):
        return [
            (0, slice(None), slice(None), slice(0, 3)),          # within first LAM
            (0, slice(None), slice(None), slice(2, total - 1)),  # spans the LAM/globe boundary
            (slice(0, 2), slice(None), slice(None), slice(total - 4, total)),  # within globe tail
            (1, slice(None), slice(None), 2),                    # single grid point (int)
            (slice(0, 3), slice(0, 1), slice(None), slice(1, total, 2)),  # strided subset
        ]

    def test_pushdown_matches_eager(self, cutout):
        total = cutout.shape[-1]
        for n in self._indices(total):
            with eager_path():
                expected = cutout[n]
            actual = two_step_read(cutout, n)
            np.testing.assert_array_equal(actual, expected, err_msg=f"index {n}")


class TestCutoutPushdownEfficiency:
    def test_reads_only_requested_points(self, cutout):
        total = cutout.shape[-1]
        lam_len = cutout.lams[0].shape[-1]

        # Full grid: the eager-style path reads every constituent grid point.
        full_parts, _ = factorize(cutout.collect_read_parts((0, slice(None), slice(None), slice(None))))
        full_points = _grid_points_read(full_parts)
        assert full_points >= total  # reads full LAM grid + full globe grid

        # A shard that stays within ONE constituent (so the other store is
        # skipped) pushes down and reads exactly K grid points.
        for k0, k1 in [(0, 3), (lam_len, lam_len + 3), (total - 4, total)]:
            n = (0, slice(None), slice(None), slice(k0, k1))
            parts, _ = factorize(cutout.collect_read_parts(n))
            assert all(p.grid_index is not None for p in parts), [str(p) for p in parts]
            assert _grid_points_read(parts) == (k1 - k0)

    def test_spanning_shard_falls_back_to_eager(self, cutout):
        # A shard that spans the LAM/globe boundary touches every store, so the
        # guard declines pushdown (no store skipped) and the eager full read is
        # used — still byte-identical, just no grid_index parts.
        lam_len = cutout.lams[0].shape[-1]
        n = (0, slice(None), slice(None), slice(lam_len - 1, lam_len + 2))
        parts, _ = factorize(cutout.collect_read_parts(n))
        assert all(p.grid_index is None for p in parts), [str(p) for p in parts]
        with eager_path():
            expected = cutout[n]
        np.testing.assert_array_equal(two_step_read(cutout, n), expected)


class TestMultiCutoutGridUnion:
    """D2: two cutouts sharing one globe, each asking a grid subset that hits the
    globe, must read the shared globe once (grid indices unioned)."""

    def _two_cutouts_sharing_globe(self):
        pytest.importorskip("anemoi.transform.spatial")
        from anemoi.datasets.usage.gridded.grids import Cutout

        globe = GriddedZarr(_group(20, np.linspace(-80, 80, 20), np.linspace(0, 350, 20)), path="globe.zarr")
        lam_a = GriddedZarr(
            _group(10, np.array([5.0, 10.0, 15.0, 5.0, 10.0]), np.array([5.0, 5.0, 5.0, 10.0, 10.0])),
            path="lam_a.zarr",
        )
        lam_b = GriddedZarr(
            _group(11, np.array([-5.0, -10.0, -15.0, -5.0, -10.0]), np.array([200.0, 200.0, 200.0, 205.0, 205.0])),
            path="lam_b.zarr",
        )
        ca = Cutout([lam_a, globe], axis=3, cropping_distance=500.0)
        cb = Cutout([lam_b, globe], axis=3, cropping_distance=500.0)
        return ca, cb, globe

    def test_shared_globe_unioned_to_one_read(self):
        from anemoi.datasets.usage.gridded.multi import Multi

        ca, cb, globe = self._two_cutouts_sharing_globe()
        ds = Multi({"a": ca, "b": cb})

        total = min(ca.shape[-1], cb.shape[-1])
        lam_len = ca.lams[0].shape[-1]
        # A shard wholly inside the globe region of both cutouts: each skips its
        # LAM and pushes the globe down, so the two globe reads can be unioned.
        n = (0, slice(None), slice(None), slice(lam_len + 1, total))

        parts, _ = factorize(ds.collect_read_parts(n))
        globe_parts = [p for p in parts if p.data is globe.data]
        assert len(globe_parts) == 1, [str(p) for p in parts]   # unioned, not 2
        assert globe_parts[0].grid_index is not None

        # Correctness: each member matches its standalone cutout.
        with eager_path():
            exp_a, exp_b = ca[n], cb[n]
        result = two_step_read(ds, n)
        np.testing.assert_array_equal(result["a"], exp_a)
        np.testing.assert_array_equal(result["b"], exp_b)


class TestPlainStoreGridIndex:
    def test_grid_list_index_uses_pushdown_and_matches(self):
        store = GriddedZarr(_group(1, np.linspace(-80, 80, 30), np.linspace(0, 350, 30)), path="s.zarr")
        # date as a slice so the eager expand_list_indexing path keeps 4 dims.
        n = (slice(0, 2), slice(None), slice(None), [2, 5, 9, 17])
        with eager_path():
            expected = store[n]
        actual = two_step_read(store, n)
        np.testing.assert_array_equal(actual, expected)

        parts, _ = factorize(store.collect_read_parts(n))
        assert len(parts) == 1
        assert parts[0].grid_index == (2, 5, 9, 17)
        assert _grid_points_read(parts) == 4

    def test_int_date_with_grid_list_two_step_only(self):
        # Two-step pushdown handles (int_date, ..., grid_list) which the eager
        # expand_list_indexing path cannot (it drops a dim then mis-concatenates).
        store = GriddedZarr(_group(1, np.linspace(-80, 80, 30), np.linspace(0, 350, 30)), path="s.zarr")
        actual = two_step_read(store, (0, slice(None), slice(None), [2, 5, 9, 17]))
        expected = np.asarray(store.data[0:1]).take([2, 5, 9, 17], axis=-1)[0]
        np.testing.assert_array_equal(actual, expected)
