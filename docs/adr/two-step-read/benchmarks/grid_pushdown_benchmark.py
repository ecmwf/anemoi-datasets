#!/usr/bin/env python
# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Benchmark / perf test for the two-step read grid-subset pushdown (ADR 3).

Reproduces the measurements in ``RESULTS.md``.  Three modes:

  # 1. Chunking survey — does the grid axis sit in one chunk? (open by NAME)
  python grid_pushdown_benchmark.py survey <name> [<name> ...]

  # 2. Real cutout — full-read parity + per-shard time/memory + store skipping
  python grid_pushdown_benchmark.py cutout --lam <name> --globe <name> [--shards 16]

  # 3. Synthetic — when does pushdown cut chunk reads? (no data access needed)
  python grid_pushdown_benchmark.py synthetic

Notes
-----
* Datasets are opened **by name** (``open_dataset("<name>")``), which resolves to
  the configured store (e.g. EWC S3) — not via a filesystem path.  Read-only.
* Eager vs two-step is toggled with ``read_parts.READ_PARTS_ENABLED`` and the two
  are **interleaved** per repetition to cancel S3-latency drift; medians reported.
"""

import argparse
import datetime
import statistics
import time
import tracemalloc

import numpy as np

import anemoi.datasets.usage.read_parts as rp
from anemoi.datasets.usage.read_parts import factorize


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _grid_points_read(parts) -> int:
    """Total grid points the factorized parts touch on the last axis."""
    total = 0
    for p in parts:
        if p.grid_index is not None:
            total += len(p.grid_index)
        else:
            s, e, t = p.slices[-1]
            total += len(range(s, e, t))
    return total


def _interleaved(ds, index, reps=8, warm=3):
    """Median wall time of eager vs two-step, interleaved per rep (cancels drift)."""
    for _ in range(warm):
        rp.READ_PARTS_ENABLED = True
        ds[index]
        rp.READ_PARTS_ENABLED = False
        ds[index]
    eager, two_step = [], []
    for _ in range(reps):
        rp.READ_PARTS_ENABLED = False
        t0 = time.perf_counter(); ds[index]; eager.append(time.perf_counter() - t0)
        rp.READ_PARTS_ENABLED = True
        t0 = time.perf_counter(); ds[index]; two_step.append(time.perf_counter() - t0)
    rp.READ_PARTS_ENABLED = True
    return statistics.median(eager), statistics.median(two_step)


def _peak_mb(ds, index, enabled, reps=3):
    rp.READ_PARTS_ENABLED = enabled
    ds[index]
    tracemalloc.start()
    for _ in range(reps):
        ds[index]
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rp.READ_PARTS_ENABLED = True
    return peak / 1e6


# --------------------------------------------------------------------------- #
# mode 1: chunking survey
# --------------------------------------------------------------------------- #
def survey(names):
    from anemoi.datasets import open_dataset

    print(f"{'dataset':<58} {'grid':>10} {'grid_chunk':>11} {'grid chunked?':>13}")
    for n in names:
        try:
            ds = open_dataset(n)
            g, gc = ds.shape[-1], ds.data.chunks[-1]
            print(f"{n[:58]:<58} {g:>10} {gc:>11} {'YES' if gc < g else 'no':>13}")
        except Exception as exc:  # noqa: BLE001
            print(f"{n[:58]:<58}  ERR {type(exc).__name__}: {str(exc)[:40]}")


# --------------------------------------------------------------------------- #
# mode 2: real cutout
# --------------------------------------------------------------------------- #
def _build_cutout(lam, globe, select_common, no_var_check):
    import contextlib
    from unittest.mock import patch

    from anemoi.datasets import open_dataset

    members = [lam, globe]
    if select_common:
        common = sorted(set(open_dataset(lam).variables) & set(open_dataset(globe).variables))
        members = [{"dataset": lam, "select": common}, {"dataset": globe, "select": common}]

    ctx = contextlib.nullcontext()
    if no_var_check:
        # Cross-source LAM/global pairs often fail the (semantic) variable-units
        # compatibility check. Irrelevant for a read-perf test: bypass it.
        ctx = patch("anemoi.datasets.usage.forwards.Combined.check_variables_compatibility", return_value=None)
    with ctx:
        return open_dataset(cutout=members, adjust="dates")


def cutout(lam, globe, n_shards=16, date=0, select_common=False, no_var_check=False):
    from anemoi.datasets.usage.gridded.indexing import index_to_slices

    t0 = time.perf_counter()
    ds = _build_cutout(lam, globe, select_common, no_var_check)
    print(f"cutout built in {time.perf_counter() - t0:.1f}s  shape={ds.shape}  "
          f"grid={ds.shape[-1]}  nvars={ds.shape[1]}  pushdown_supported={ds._pushdown_supported}")
    print(f"execute_parts: sequential, no thread arg "
          f"(READ_PARTS_THREADS present: {hasattr(rp, 'READ_PARTS_THREADS')})")

    total = ds.shape[-1]
    lam_len = ds.lams[0].shape[-1]
    W = total // n_shards
    shards = {
        "FULL grid": slice(None),
        f"1/{n_shards} in LAM": slice(0, W),
        f"1/{n_shards} in globe": slice(total - W, total),
        f"1/{n_shards} spanning": slice(lam_len - W // 2, lam_len + W // 2),
    }

    print(f"\n{'access':<22} {'eager ms':>10} {'2step ms':>10} {'ratio':>7} "
          f"{'stores':>7} {'grid pts':>9} {'legMB':>7} {'2stepMB':>8}")
    for label, sh in shards.items():
        idx = (date, slice(None), slice(None), sh)
        norm, _ = index_to_slices(idx, ds.shape)
        plan = None if ds._is_full_grid_slice(norm[3]) else ds._grid_pushdown(norm, norm[3])
        parts, _ = factorize(ds.collect_read_parts(idx))
        stores = len({id(p.data) for p in parts})
        leg, two = _interleaved(ds, idx)
        lmb = _peak_mb(ds, idx, False)
        tmb = _peak_mb(ds, idx, True)
        pts = _grid_points_read(parts) if plan is not None else total
        print(f"{label:<22} {leg*1e3:>10.0f} {two*1e3:>10.0f} {leg/two:>7.2f} "
              f"{stores:>7} {pts:>9} {lmb:>7.0f} {tmb:>8.0f}")


# --------------------------------------------------------------------------- #
# mode 3: synthetic (counts real chunk reads)
# --------------------------------------------------------------------------- #
class _CountingStore:
    pass


def synthetic():
    import zarr

    class CountingStore(zarr.storage.KVStore):
        def __init__(self, inner):
            super().__init__(inner)
            self.reset()

        def reset(self):
            self.reads = self.bytes = 0

        def __getitem__(self, key):
            v = super().__getitem__(key)
            if key.startswith("data/") and ".z" not in key.rsplit("/", 1)[-1]:
                self.reads += 1
                self.bytes += len(v)
            return v

    def build(n_dates, n_vars, n_grid, lats, lons, grid_chunk, seed):
        from anemoi.datasets.usage.gridded.store import GriddedZarr

        counting = CountingStore(zarr.storage.MemoryStore())
        root = zarr.open_group(store=counting, mode="w")
        data = np.random.default_rng(seed).standard_normal((n_dates, n_vars, 1, n_grid)).astype(np.float32)
        root.create_dataset("data", data=data, chunks=(1, n_vars, 1, grid_chunk), compressor=None, overwrite=True)
        freq = datetime.timedelta(hours=6)
        dates = np.array([datetime.datetime(2021, 1, 1) + i * freq for i in range(n_dates)], dtype="datetime64")
        root.create_dataset("dates", data=dates, overwrite=True)
        root.create_dataset("latitudes", data=lats, overwrite=True)
        root.create_dataset("longitudes", data=lons, overwrite=True)
        for nm in ("mean", "stdev", "maximum", "minimum"):
            root.create_dataset(nm, data=np.zeros(n_vars), overwrite=True)
        root.attrs.update({
            "frequency": "6h", "resolution": "o96",
            "name_to_index": {f"v{i}": i for i in range(n_vars)},
            "data_request": {"grid": 1, "area": "g", "param_level": {}},
            "variables_metadata": {f"v{i}": {} for i in range(n_vars)},
            "field_shape": [1, n_grid],
        })
        return GriddedZarr(root, path="<mem>"), counting

    from anemoi.datasets.usage.gridded.grids import Cutout

    def run(grid_chunk, label):
        n_glob, n_lam, n_dates, n_vars = 20000, 2000, 4, 8
        lam, lam_s = build(n_dates, n_vars, n_lam,
                           np.random.default_rng(0).uniform(40, 50, n_lam),
                           np.random.default_rng(1).uniform(0, 10, n_lam), min(grid_chunk, n_lam), 10)
        globe, globe_s = build(n_dates, n_vars, n_glob,
                               np.random.default_rng(2).uniform(-90, 90, n_glob),
                               np.random.default_rng(3).uniform(0, 360, n_glob), grid_chunk, 20)
        ds = Cutout([lam, globe], axis=3, cropping_distance=500.0)
        total = ds.shape[-1]
        n = (0, slice(None), slice(None), slice(total - total // 8, total))  # 1/8 in globe region

        def chunks(enabled):
            lam_s.reset(); globe_s.reset()
            rp.READ_PARTS_ENABLED = enabled
            ds[n]
            rp.READ_PARTS_ENABLED = True
            return lam_s.reads + globe_s.reads, lam_s.bytes + globe_s.bytes

        lr, lb = chunks(False)
        pr, pb = chunks(True)
        print(f"\n[{label}] grid_chunk={grid_chunk}  globe={n_glob} lam={n_lam}  "
              f"shard=1/8 in globe ({total - (total - total // 8)} pts)")
        print(f"  eager  : {lr:3d} chunk reads / {lb:>10,} B")
        print(f"  pushdown: {pr:3d} chunk reads / {pb:>10,} B  ({100 * pb / lb:.0f}% of eager)")

    print("=== SYNTHETIC: grid pushdown chunk reads vs eager ===")
    run(grid_chunk=20000, label="whole grid in ONE chunk (anemoi default)")
    run(grid_chunk=1000, label="grid SPLIT into chunks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("survey"); s.add_argument("names", nargs="+")
    c = sub.add_parser("cutout")
    c.add_argument("--lam", required=True); c.add_argument("--globe", required=True)
    c.add_argument("--shards", type=int, default=16); c.add_argument("--date", type=int, default=0)
    c.add_argument("--select-common", action="store_true",
                   help="select the variables common to both members (needed for real cross-source pairs)")
    c.add_argument("--no-var-check", action="store_true",
                   help="bypass the semantic variable-units compatibility check (read-perf only)")
    sub.add_parser("synthetic")
    args = ap.parse_args()

    if args.mode == "survey":
        survey(args.names)
    elif args.mode == "cutout":
        cutout(args.lam, args.globe, n_shards=args.shards, date=args.date,
               select_common=args.select_common, no_var_check=args.no_var_check)
    elif args.mode == "synthetic":
        synthetic()
