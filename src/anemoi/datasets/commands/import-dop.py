# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import heapq
import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from threading import Lock
from typing import Any

import numpy as np
import tqdm
import zarr

from anemoi.datasets.tabular.btree import ZarrBTree
from anemoi.datasets.tabular.caching import ChunksCache

from . import Command

LOG = logging.getLogger(__name__)

FACTORS = {
    "datetime64[s]": 1,
    "datetime64[ms]": 1_000,
    "datetime64[us]": 1_000_000,
    "datetime64[ns]": 1_000_000_000,
}


def _duplicate_ranges(a):
    if a.size == 0:
        return []

    # True where value differs from previous => boundaries
    boundaries = np.r_[True, a[1:] != a[:-1], True]

    # Indices where boundaries occur
    idx = np.flatnonzero(boundaries)

    # Pair successive boundaries → (start,end) for each group
    ranges = list(zip(idx[:-1], idx[1:]))

    return [(s, e - s) for s, e in ranges]


class ImportDOP(Command):
    """Import DOP datasets."""

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser.
        """

        command_parser.add_argument("--tmpdir", metavar="TMPDIR", default=tempfile.gettempdir())
        command_parser.add_argument("--max-workers", type=int, default=20)
        command_parser.add_argument("input_path", metavar="INPUT_DATASET")
        command_parser.add_argument("output_path", metavar="OUTPUT_DATASET")

    def run(self, args: Any) -> None:

        self.input_path = args.input_path
        self.output_path = args.output_path
        self.tmpdir = args.tmpdir
        self.max_workers = args.max_workers

        os.makedirs(self.tmpdir, exist_ok=True)

        self.pass1_path = os.path.join(self.tmpdir, "pass1.mmap")
        self.pass2_path = os.path.join(self.tmpdir, "pass2.done")
        self.pass3_path = os.path.join(self.tmpdir, "pass3.done")

        self.input_zarr = zarr.open(self.input_path, mode="r")

        try:
            # If the output Zarr already exists and is valid, call cleanup()
            z = zarr.open(self.output_path, mode="r")
            if "data" not in z:
                self.cleanup()
        except Exception:
            pass

        self.output_zarr = zarr.open(self.output_path, mode="w")
        self.output_zarr.attrs["format"] = "tabular"

        self.all_dates = self.input_zarr["dates"]
        self.all_data = self.input_zarr["data"]

        self.factor = FACTORS[str(self.all_dates.dtype)]

        self.btree = ZarrBTree(self.output_zarr, mode="a")

        LOG.info(
            f"{self.all_dates.shape[0]:,} dates in total, chunks: {self.all_dates.chunks}"
            f" block size: {math.prod(self.all_dates.chunks) * self.all_dates.dtype.itemsize} bytes,"
            f" {math.ceil(self.all_dates.shape[0]/(self.all_dates.chunks[0])):,} chunks"
        )

        LOG.info(
            f"{self.all_data.shape[0]:,} data in total, chunks: {self.all_data.chunks}"
            f" block size: {math.prod(self.all_data.chunks) * self.all_data.dtype.itemsize} bytes,"
            f" {math.ceil(self.all_data.shape[0]/(self.all_data.chunks[0])):,} chunks"
        )

        self.chunk_rows = 1_000_000
        self.out_shape = (
            self.all_data.shape[0],
            self.all_data.shape[1] + 2,
        )  # +2 for date columns

        self.pre_checks()

        self.pass1()
        self.pass2()
        self.pass3()

        self.cleanup()

        self.post_checks()

    def pass1(self):

        if os.path.exists(self.pass1_path):
            LOG.info(f"Pass 1 memmap already exists at {self.pass1_path}, skipping...")
            return

        if os.path.exists(self.pass2_path):
            os.remove(self.pass2_path)

        if os.path.exists(self.pass3_path):
            os.remove(self.pass3_path)

        step = self.all_data.chunks[0]
        count = math.ceil(len(self.all_dates) / step)

        lat_idx = self.all_data.attrs["colnames"].index("lat")
        lon_idx = self.all_data.attrs["colnames"].index("lon")
        # Move 'lat' and 'lon' columns to the beginning of data
        cols = list(range(self.all_data.shape[1]))
        new_order = [lat_idx, lon_idx] + [i for i in cols if i not in (lat_idx, lon_idx)]

        self.output_zarr.attrs["columns"] = [
            "_days_since_epoch",
            "_seconds_since_midnight",
            "_latitudes",
            "_longitudes",
        ] + [c for c in self.all_data.attrs["colnames"] if c not in ("lat", "lon")]

        tmp_pass1_path = self.pass1_path + ".tmp"

        out = np.memmap(tmp_pass1_path, dtype="float32", mode="w+", shape=self.out_shape)
        LOG.info(
            f"Output memmap created at {tmp_pass1_path} with shape {self.out_shape}"
            f" and dtype float32 {os.path.getsize(tmp_pass1_path):,} bytes"
        )

        def worker(n, step):
            end = min(n + step, self.all_dates.shape[0])
            dates = self.all_dates[n:end].flatten()
            data = self.all_data[n:end]

            x = np.round(self.all_dates[n : n + step].flatten().astype("int64") / self.factor)

            dt = x // 86400
            tm = x % 86400
            dates = np.column_stack([dt.astype("float32"), tm.astype("float32")])

            data = data[:, new_order]
            data = np.concatenate([dates, data], axis=1)

            out[n:end] = data

        tasks = []
        n = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            LOG.info(f"Starting to load and swap columns and add date columns... {count:,} chunks to process")
            for _ in tqdm.tqdm(range(count)):
                tasks.append(executor.submit(worker, n, step))
                n += step

            LOG.info(f"Waiting for {len(tasks):,} swap columns and add date columns tasks to complete...")
            for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), smoothing=0):
                future.result()

        out.flush()
        os.rename(tmp_pass1_path, self.pass1_path)

    def pass2(self):
        """External lexicographic sort of a very large (N, K) array stored in .npy format.
        Lexicographic order is on columns 0..K-1.
        """

        if os.path.exists(self.pass2_path):
            LOG.info(f"Pass 2 already done at {self.pass2_path}, skipping...")
            return

        if os.path.exists(self.pass3_path):
            os.remove(self.pass3_path)

        arr = np.memmap(self.pass1_path, mode="r", dtype="float32", shape=self.out_shape)
        N, K = arr.shape

        temp_files = []

        tasks = []

        def worker_chunk_sort(arr, start, chunk_rows, N):
            end = min(start + chunk_rows, N)
            chunk = np.array(arr[start:end])  # load into memory

            # numeric lexsort: primary key = col0, then col1, ...
            idx = np.lexsort(chunk.T[::-1])
            chunk = chunk[idx]

            tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.tmpdir, suffix=".npy")
            np.save(tmp, chunk)
            tmp.close()
            return tmp.name

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            LOG.info(f"chunk sort phase: Starting to sort chunks... {math.ceil(N/self.chunk_rows):,} chunks to process")
            # --- PHASE 1: chunk sort ---
            for start in tqdm.tqdm(range(0, N, self.chunk_rows)):
                tasks.append(executor.submit(worker_chunk_sort, arr, start, self.chunk_rows, N))

            LOG.info(f"Waiting for {len(tasks):,} tasks to complete...")
            for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), smoothing=0):
                temp_files.append(future.result())

        # --- PHASE 2: merge sorted chunks ---
        fps = [np.load(p, mmap_mode="r") for p in temp_files]

        # Convert each sorted chunk into an iterator of (key_tuple, row)
        def keyed_iter(a):
            for row in a:
                yield (tuple(row), row)

        merged = heapq.merge(*(keyed_iter(fp) for fp in fps))

        LOG.info(f"Input array size: {arr.size * arr.dtype.itemsize:,} bytes")
        row_size = arr.shape[1] * arr.dtype.itemsize
        chunk = round((256 * 1024 * 1024) / row_size)  # 256 MB chunks
        LOG.info(f"Output chunk size: {(chunk, self.out_shape[1])}")

        # write output (same shape)
        self.output_zarr.create_dataset(
            "data",
            shape=self.out_shape,
            chunks=(chunk, self.out_shape[1]),
            dtype="float32",
            fill_value=np.nan,
            # compressor=None,
        )
        self.output_zarr["data"].attrs["_ARRAY_DIMENSIONS"] = ["dates", "columns"]
        out = ChunksCache(self.output_zarr["data"])
        i = 0
        prev = None
        for _, row in tqdm.tqdm(merged, total=N, smoothing=0):
            if prev is not None and np.array_equal(prev, row):
                raise ValueError("Duplicate rows detected during merge")
            prev = row
            out[i] = row
            i += 1
        out.flush()

        # cleanup
        for p in tqdm.tqdm(temp_files):
            os.remove(p)

        with open(self.pass2_path, "w"):
            pass

    def pass3(self):
        lock = Lock()
        step = self.all_data.chunks[0]
        count = math.ceil(len(self.all_dates) / step)
        tasks = []
        prev = None
        off = 0
        LOG.info(f"Starting to load B-tree... {count:,} chunks to process")
        n = 0
        if os.path.exists(self.pass3_path):
            LOG.info(f"Pass 3 already done at {self.pass3_path}, skipping...")
            return

        def set_btree(key, value):
            with lock:
                self.btree.set(key, value)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for _ in tqdm.tqdm(range(count)):
                x = np.round(self.all_dates[n : n + step].flatten().astype("int64") / self.factor)
                if prev is not None:
                    x = np.concatenate((prev, x))
                offset = n - off

                ranges = _duplicate_ranges(x)
                last = ranges.pop()

                for s, e in ranges:
                    tasks.append(executor.submit(set_btree, x[s], (s + offset, e)))
                    # set_btree_raw(x[s], (s + offset, e))

                prev = x[last[0] :]
                assert len(prev) > 0, (last, len(x), ranges)
                off = len(prev)
                n += step

            assert last[0] + offset + last[1] == len(self.all_dates), (
                last,
                last[0] + offset,
                len(prev),
                off,
                len(self.all_dates),
                len(x),
            )

            tasks.append(executor.submit(set_btree, prev[0], (last[0] + offset, last[1])))
            LOG.info(f"Waiting for {len(tasks):,} b-tree tasks to complete...")
            for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), smoothing=0):
                future.result()

        self.btree.flush()
        LOG.info(f"{self.btree.height()=}, {self.btree.size()=}")
        with open(self.pass3_path, "w"):
            pass

    def cleanup(self):
        if os.path.exists(self.pass1_path):
            os.remove(self.pass1_path)

        if os.path.exists(self.pass2_path):
            os.remove(self.pass2_path)

        if os.path.exists(self.pass3_path):
            os.remove(self.pass3_path)

    def pre_checks(self):
        assert self.input_zarr["dates"].shape[0] == self.input_zarr["data"].shape[0], (
            self.input_zarr["data"].shape,
            self.input_zarr["dates"].shape,
        )

        assert len(self.input_zarr["data"].shape) == 2, self.input_zarr["data"].shape

    def post_checks(self):

        assert len(self.output_zarr["data"].shape) == 2, self.output_zarr["data"].shape

        assert self.output_zarr["data"].shape[0] == self.input_zarr["data"].shape[0], (
            self.output_zarr["data"].shape[0],
            self.input_zarr["data"].shape[0],
        )

        assert self.output_zarr["data"].shape[1] == self.input_zarr["data"].shape[1] + 2, (
            self.output_zarr["data"].shape[1],
            self.input_zarr["data"].shape[1],
        )

        assert self.btree.count() == self.output_zarr["data"].shape[0], (
            self.btree.count(),
            self.output_zarr["data"].shape[0],
        )
