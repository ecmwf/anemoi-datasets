# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os

# from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import ProcessPoolExecutor as Executor
from concurrent.futures import as_completed

import numpy as np
import tqdm

from anemoi.datasets.tabular.caching import ChunksCache

LOG = logging.getLogger(__name__)


class Chunk:
    def __init__(self, /, first_date, last_date, shape, file_path):
        self.file_path = file_path
        self.first_date = first_date
        self.last_date = last_date
        self.shape = shape

    @classmethod
    def from_array(cls, array: np.ndarray, file_path: str):
        if len(array) == 0:
            return None
        first_date = _date(array, 0)
        last_date = _date(array, -1)
        shape = array.shape
        return cls(first_date=first_date, last_date=last_date, shape=shape, file_path=file_path)

    @classmethod
    def from_path(cls, file_path: str):
        array = np.load(file_path, mmap_mode="r")
        return cls.from_array(array, file_path=file_path)


def _unduplicate_rows(array: np.ndarray) -> np.ndarray:
    """Remove duplicate rows from a 2D numpy array, handling NaNs correctly."""
    assert len(array.shape) == 2, f"Expected 2D array, got shape {array.shape}"
    # Remove duplicate rows. np.unique does not work well with NaNs, so we replace them with a sentinel value.

    b = np.ascontiguousarray(array)
    b2 = np.nan_to_num(b, nan=np.inf)

    row_dtype = np.dtype((np.void, b2.dtype.itemsize * b2.shape[1]))
    _, idx = np.unique(b2.view(row_dtype), return_index=True)

    unique_array = b[np.sort(idx)]

    duplicates = len(array) - len(unique_array)
    return duplicates, unique_array


def _date(array: np.ndarray, index: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(int(array[index][0]) * 86400 + int(array[index][1]))


def _unduplicate_worker(file_path: str):

    array = np.load(file_path, mmap_mode="r")

    duplicates, unique_array = _unduplicate_rows(array)

    if duplicates:
        # LOG.warning(f"Removed {duplicates} duplicate rows from {file_path}")
        np.save(file_path + ".tmp", unique_array)  # Save to a temporary file first, numpy will add a .npy extension
        os.rename(file_path + ".tmp.npy", file_path)

    first_date = _date(unique_array, 0)
    last_date = _date(unique_array, -1)

    return duplicates, Chunk(first_date=first_date, last_date=last_date, shape=unique_array.shape, file_path=file_path)


def _deoverlap_worker(one, two):
    array_one = np.load(one.file_path, mmap_mode="r")
    array_two = np.load(two.file_path, mmap_mode="r")

    i = 0
    last = _date(array_one, -1)
    while last >= _date(array_two, i) and i < array_two.shape[0]:
        i += 1

    assert i > 0

    duplicates = 0

    beginning_array_two = array_two[:i]

    if np.allclose(array_one[-1], beginning_array_two[0], equal_nan=True, rtol=0, atol=0):
        duplicates = 1
        beginning_array_two = beginning_array_two[1:]

    new_array_one = np.vstack([array_one, beginning_array_two])
    end_array_two = array_two[i:]

    np.save(one.file_path + ".tmp", new_array_one)
    os.rename(one.file_path + ".tmp.npy", one.file_path)

    np.save(two.file_path + ".tmp", end_array_two)
    os.rename(two.file_path + ".tmp.npy", two.file_path)

    result = []

    if new_array_one.size > 0:
        one = Chunk.from_path(one.file_path)
        assert one is not None
        result.append(one)
    else:
        os.unlink(one.file_path)

    if end_array_two.size > 0:
        two = Chunk.from_path(two.file_path)
        assert two is not None
        result.append(two)
    else:
        os.unlink(two.file_path)

    return duplicates, result


def _find_duplicate_and_overlapping_dates(work_dir: str, max_workers: int = None):

    import os

    files = [
        os.path.join(work_dir, file) for file in os.listdir(work_dir) if file.endswith(".npy") and ".tmp" not in file
    ]

    chunks = {}
    total_duplicates = 0

    if max_workers is None:
        max_workers = min(os.cpu_count(), 20)

    with Executor(max_workers=max_workers) as executor:

        tasks = []
        for file in files:
            tasks.append(executor.submit(_unduplicate_worker, file))

        LOG.info("Checking duplicates")
        for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), desc="Checking duplicates", unit="file"):
            duplicates, chunk = future.result()
            total_duplicates += duplicates
            chunks[chunk.file_path] = chunk

        LOG.info("Done")

        while True:

            tasks = []
            prev = None
            for chunk in sorted(chunks.values(), key=lambda x: x.first_date):
                if prev is None:
                    prev = chunk
                    continue

                if chunk.first_date <= prev.last_date:
                    tasks.append(executor.submit(_deoverlap_worker, prev, chunk))
                    del chunks[prev.file_path]
                    del chunks[chunk.file_path]
                    prev = None
                else:
                    prev = chunk

            if not tasks:
                break

            LOG.info("Checking overlaps")
            for future in tqdm.tqdm(as_completed(tasks), total=len(tasks), desc="Checking overlaps", unit="pair"):
                duplicates, updates = future.result()
                total_duplicates += duplicates
                chunks.update({update.file_path: update for update in updates})

    LOG.info(f"Total duplicates removed: {total_duplicates}")
    return sorted(chunks.values(), key=lambda x: x.first_date)


def finalise_tabular_dataset(*, store, work_dir: str, delete_files: bool = True, max_workers: int = None):
    chunks = _find_duplicate_and_overlapping_dates(work_dir, max_workers=max_workers)
    assert chunks, "No data found to finalise"
    shape = chunks[0].shape
    assert all(chunk.shape[1] == shape[1] for chunk in chunks), "Inconsistent number of columns in chunks"
    shape = (sum(chunk.shape[0] for chunk in chunks), chunks[0].shape[1])

    if "data" in store:
        del store["data"]

    row_size = shape[1] * np.dtype(np.float32).itemsize
    target_size = 64 * 1024 * 1024  # Target chunk size of 64 MB
    chunk_size = max(1, round(target_size / row_size))

    chunking = (min(chunk_size, shape[0]), shape[1])
    LOG.info(f"Final dataset shape: {shape}, chunking: {chunking}")
    store.create_dataset(
        "data",
        shape=shape,
        chunks=chunking,
        dtype=np.float32,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    )
    offset = 0

    with ChunksCache(store["data"]) as data:
        with tqdm.tqdm(total=len(data), desc="Writing to Zarr", unit="row") as pbar:
            for chunk in chunks:
                array = np.load(chunk.file_path, mmap_mode="r")
                # assert array.shape == chunk.shape, f"Chunk shape mismatch: expected {chunk.shape}, got {array.shape}"
                last = offset + chunk.shape[0]
                data[offset:last, :] = array
                offset = last
                pbar.update(chunk.shape[0])


if __name__ == "__main__":
    import sys

    import zarr

    logging.basicConfig(level=logging.INFO)
    work_dir = sys.argv[2]
    store = zarr.open(sys.argv[1], mode="w")
    finalise_tabular_dataset(store=store, work_dir=work_dir, delete_files=False)
