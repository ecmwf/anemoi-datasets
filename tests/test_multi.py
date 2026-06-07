# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``open_dataset(multi=dict(...))`` and shared two-step reads.

Two things are checked:

1. **Correctness** — ``multi[n]`` returns ``{name: member[n]}`` and each value
   equals what the standalone member would return.
2. **Usefulness** — when several members share one physical store (the classic
   "several cutouts over the same global model" case), the store is read *once*.
   We prove it by counting the factorized read parts: a shared global gives
   ``N_lams + 1`` reads, whereas independent opens give ``2 * N_lams``.
"""

import datetime

import numpy as np
import pytest
import zarr

from anemoi.datasets.usage.gridded.multi import Multi
from anemoi.datasets.usage.gridded.store import GriddedZarr
from anemoi.datasets.usage.read_parts import factorize
from anemoi.datasets.usage.read_parts import two_step_read

N_DATES = 4
FREQ = datetime.timedelta(hours=6)
DATES = np.array([datetime.datetime(2021, 1, 1) + i * FREQ for i in range(N_DATES)], dtype="datetime64")


def _build_group(root, data, lats, lons):
    n_vars = data.shape[1]
    root.create_array("data", data=data, compressor=None)
    root.create_array("dates", data=DATES, compressor=None)
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
        "field_shape": [1, data.shape[-1]],
    })
    return root


def _lam_group(seed, lats, lons):
    n = len(lats)
    data = np.random.default_rng(seed).standard_normal((N_DATES, 2, 1, n)).astype(np.float32)
    return _build_group(zarr.group(), data, lats, lons)


def _globe_group(seed=20, n=20):
    data = np.random.default_rng(seed).standard_normal((N_DATES, 2, 1, n)).astype(np.float32)
    lats = np.linspace(-80, 80, n)
    lons = np.linspace(0, 350, n)
    return _build_group(zarr.group(), data, lats, lons)


def _make_cutout(lam_store, globe_store):
    from anemoi.datasets.usage.gridded.grids import Cutout

    return Cutout([lam_store, globe_store], axis=3, cropping_distance=500.0)


@pytest.fixture()
def shared_globe_cutouts():
    """Two cutouts (different LAMs) over a single shared global store object."""
    pytest.importorskip("anemoi.transform.spatial")

    globe = GriddedZarr(_globe_group(), path="globe.zarr")

    lam_a = GriddedZarr(
        _lam_group(10, np.array([5.0, 10.0, 15.0, 5.0, 10.0]), np.array([5.0, 5.0, 5.0, 10.0, 10.0])),
        path="lam_a.zarr",
    )
    lam_b = GriddedZarr(
        _lam_group(11, np.array([-5.0, -10.0, -15.0, -5.0, -10.0]), np.array([200.0, 200.0, 200.0, 205.0, 205.0])),
        path="lam_b.zarr",
    )

    cutout_a = _make_cutout(lam_a, globe)
    cutout_b = _make_cutout(lam_b, globe)
    return cutout_a, cutout_b, globe


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


class TestMultiCorrectness:
    def test_returns_named_dict(self, shared_globe_cutouts):
        cutout_a, cutout_b, _ = shared_globe_cutouts
        ds = Multi({"a": cutout_a, "b": cutout_b})

        for n in [0, 2, slice(0, 3), (slice(0, 2), slice(None), slice(None), slice(None))]:
            result = ds[n]
            assert set(result) == {"a", "b"}
            np.testing.assert_array_equal(result["a"], cutout_a[n])
            np.testing.assert_array_equal(result["b"], cutout_b[n])

    def test_two_step_matches_eager(self, shared_globe_cutouts):
        cutout_a, cutout_b, _ = shared_globe_cutouts
        ds = Multi({"a": cutout_a, "b": cutout_b})

        for n in [0, 1, 2, 3, slice(1, 4)]:
            via_two_step = two_step_read(ds, n)
            eager = Multi.__getitem__(ds, n)  # bypass the gate, force eager
            for k in via_two_step:
                np.testing.assert_array_equal(via_two_step[k], eager[k])

    def test_single_member(self, shared_globe_cutouts):
        cutout_a, _, _ = shared_globe_cutouts
        ds = Multi({"only": cutout_a})
        np.testing.assert_array_equal(ds[0]["only"], cutout_a[0])

    def test_length_mismatch_rejected(self, shared_globe_cutouts):
        cutout_a, _, _ = shared_globe_cutouts
        from anemoi.datasets.usage.gridded.subset import Subset

        half = Subset(cutout_a, list(range(len(cutout_a) // 2)), reason={})
        with pytest.raises(ValueError, match="same length"):
            Multi({"a": cutout_a, "b": half})

    def test_supporting_arrays_namespaced_by_member(self, shared_globe_cutouts):
        cutout_a, cutout_b, _ = shared_globe_cutouts
        ds = Multi({"a": cutout_a, "b": cutout_b})
        collected = []
        ds.collect_supporting_arrays(collected)
        paths = {c[0] for c in collected}
        # Each cutout's masks land under its multi key.
        assert any(p[0] == "a" for p in paths)
        assert any(p[0] == "b" for p in paths)
        assert all(c[1] == "cutout_mask" for c in collected)


# ---------------------------------------------------------------------------
# Usefulness: the shared global store is read once
# ---------------------------------------------------------------------------


class TestMultiSharesReads:
    def test_shared_globe_read_once(self, shared_globe_cutouts):
        cutout_a, cutout_b, globe = shared_globe_cutouts
        ds = Multi({"a": cutout_a, "b": cutout_b})

        parts = ds.collect_read_parts(0)
        merged, _ = factorize(parts)

        # 2 LAMs + 1 shared globe = 3 physical reads (not 4).
        assert len(merged) == 3, [str(p) for p in merged]

        # Exactly one merged part points at the globe store's data array.
        globe_reads = [m for m in merged if m.data is globe.data]
        assert len(globe_reads) == 1

    def test_independent_opens_do_not_share(self):
        """Contrast: cutouts over *separately opened* globes do not dedup."""
        pytest.importorskip("anemoi.transform.spatial")

        globe_1 = GriddedZarr(_globe_group(seed=20), path="globe.zarr")
        globe_2 = GriddedZarr(_globe_group(seed=20), path="globe.zarr")  # same data, different object
        lam_a = GriddedZarr(
            _lam_group(10, np.array([5.0, 10.0, 15.0, 5.0, 10.0]), np.array([5.0, 5.0, 5.0, 10.0, 10.0])),
            path="lam_a.zarr",
        )
        lam_b = GriddedZarr(
            _lam_group(11, np.array([-5.0, -10.0, -15.0, -5.0, -10.0]), np.array([200.0, 200.0, 200.0, 205.0, 205.0])),
            path="lam_b.zarr",
        )
        cutout_a = _make_cutout(lam_a, globe_1)
        cutout_b = _make_cutout(lam_b, globe_2)

        parts = cutout_a.collect_read_parts(0) + cutout_b.collect_read_parts(0)
        merged, _ = factorize(parts)

        # 2 LAMs + 2 distinct globe objects = 4 reads — the saving multi unlocks.
        assert len(merged) == 4


# ---------------------------------------------------------------------------
# End-to-end via open_dataset + shared_zarr_opens (on-disk stores)
# ---------------------------------------------------------------------------


def _write_zarr(path, group):
    """Persist an in-memory zarr group to a directory store at *path*."""
    out = zarr.open(str(path), mode="w")
    zarr.copy_all(group, out)
    return str(path)


class TestOpenDatasetMulti:
    def test_shared_open_via_factory(self, tmp_path):
        pytest.importorskip("anemoi.transform.spatial")
        from anemoi.datasets import open_dataset

        globe_path = _write_zarr(tmp_path / "globe.zarr", _globe_group())
        lam_a_path = _write_zarr(
            tmp_path / "lam_a.zarr",
            _lam_group(10, np.array([5.0, 10.0, 15.0, 5.0, 10.0]), np.array([5.0, 5.0, 5.0, 10.0, 10.0])),
        )
        lam_b_path = _write_zarr(
            tmp_path / "lam_b.zarr",
            _lam_group(11, np.array([-5.0, -10.0, -15.0, -5.0, -10.0]), np.array([200.0, 200.0, 200.0, 205.0, 205.0])),
        )

        ds = open_dataset(
            multi=dict(
                a={"cutout": [lam_a_path, globe_path], "cropping_distance": 500.0},
                b={"cutout": [lam_b_path, globe_path], "cropping_distance": 500.0},
            )
        )

        # The factory opened the shared globe once: factorize collapses to 3 reads.
        parts = ds.collect_read_parts(0)
        merged, _ = factorize(parts)
        assert len(merged) == 3, [str(p) for p in merged]

        # Correctness vs standalone cutouts.
        cutout_a = open_dataset(cutout=[lam_a_path, globe_path], cropping_distance=500.0)
        cutout_b = open_dataset(cutout=[lam_b_path, globe_path], cropping_distance=500.0)
        result = ds[0]
        np.testing.assert_array_equal(result["a"], cutout_a[0])
        np.testing.assert_array_equal(result["b"], cutout_b[0])


class TestMultiIndexingSemantics:
    """Pin the per-member indexing contract (members differ on var/grid axes)."""

    def _multi(self):
        # Two members with DIFFERENT grid sizes, same dates.
        a = GriddedZarr(_globe_group(seed=1, n=10), path="a.zarr")
        b = GriddedZarr(_globe_group(seed=2, n=6), path="b.zarr")
        return Multi({"a": a, "b": b}), a, b

    def test_date_index_returns_per_member_full_fields(self):
        ds, a, b = self._multi()
        for n in [0, slice(0, 2)]:
            res = ds[n]
            np.testing.assert_array_equal(res["a"], a[n])
            np.testing.assert_array_equal(res["b"], b[n])
            # different grid widths preserved per member
            assert res["a"].shape[-1] == 10
            assert res["b"].shape[-1] == 6

    def test_grid_slice_applied_per_member(self):
        ds, a, b = self._multi()
        # slice valid for both
        n = (0, slice(None), slice(None), slice(0, 5))
        res = ds[n]
        np.testing.assert_array_equal(res["a"], a[n])
        np.testing.assert_array_equal(res["b"], b[n])
        # slice past b's grid (6): a -> 2 points, b -> empty (numpy-like, per member)
        n2 = (0, slice(None), slice(None), slice(8, 12))
        res2 = ds[n2]
        assert res2["a"].shape[-1] == 2
        assert res2["b"].shape[-1] == 0


class TestMultiDictIndex:
    """Per-member index dict: ds[{name: index}] indexes each member by its own index."""

    def _multi(self):
        a = GriddedZarr(_globe_group(seed=1, n=10), path="a.zarr")
        b = GriddedZarr(_globe_group(seed=2, n=6), path="b.zarr")
        return Multi({"a": a, "b": b}), a, b

    def test_dict_indexes_each_member_independently(self):
        ds, a, b = self._multi()
        ia = (0, slice(None), slice(None), slice(0, 4))   # 4 grid pts from a
        ib = (0, slice(None), slice(None), slice(2, 6))   # 4 grid pts from b
        res = ds[{"a": ia, "b": ib}]
        assert set(res) == {"a", "b"}
        np.testing.assert_array_equal(res["a"], a[ia])
        np.testing.assert_array_equal(res["b"], b[ib])

    def test_dict_two_step_matches_eager(self):
        ds, a, b = self._multi()
        import anemoi.datasets.usage.read_parts as rp
        idx = {"a": (slice(0, 2), slice(None), slice(None), slice(1, 5)),
               "b": (slice(0, 2), slice(None), slice(None), slice(0, 3))}
        rp.READ_PARTS_ENABLED = False
        eager = ds[idx]
        rp.READ_PARTS_ENABLED = True
        two = ds[idx]
        for k in ("a", "b"):
            np.testing.assert_array_equal(two[k], eager[k])

    def test_dict_subset_reads_only_named_members(self):
        ds, a, b = self._multi()
        res = ds[{"a": 0}]
        assert set(res) == {"a"}
        np.testing.assert_array_equal(res["a"], a[0])

    def test_dict_unknown_key_raises(self):
        ds, a, b = self._multi()
        with pytest.raises(KeyError, match="unknown member"):
            ds[{"a": 0, "zzz": 0}]

    def test_dict_shared_store_read_once(self):
        """Two cutouts sharing a globe, each with its own grid shard, read the
        globe once (union factorization)."""
        pytest.importorskip("anemoi.transform.spatial")
        globe = GriddedZarr(_globe_group(), path="globe.zarr")
        lam_a = GriddedZarr(_lam_group(10, np.array([5.0, 10.0, 15.0, 5.0, 10.0]),
                                       np.array([5.0, 5.0, 5.0, 10.0, 10.0])), path="lam_a.zarr")
        lam_b = GriddedZarr(_lam_group(11, np.array([-5.0, -10.0, -15.0, -5.0, -10.0]),
                                       np.array([200.0, 200.0, 200.0, 205.0, 205.0])), path="lam_b.zarr")
        ca, cb = _make_cutout(lam_a, globe), _make_cutout(lam_b, globe)
        ds = Multi({"a": ca, "b": cb})
        # globe-region shard for each (skips its LAM), different per member
        la, lb = ca.lams[0].shape[-1], cb.lams[0].shape[-1]
        idx = {"a": (0, slice(None), slice(None), slice(la + 1, ca.shape[-1])),
               "b": (0, slice(None), slice(None), slice(lb + 1, cb.shape[-1]))}
        parts, _ = factorize(ds.collect_read_parts(idx))
        globe_parts = [p for p in parts if p.data is globe.data]
        assert len(globe_parts) == 1  # unioned, read once
