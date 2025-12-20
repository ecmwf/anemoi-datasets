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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np
import tqdm

from anemoi.datasets.tabular.caching import ChunksCache

LOG = logging.getLogger(__name__)
MMAP_KWARGS = {"mmap_mode": "r"}
# MMAP_KWARGS = {}


class Chunk:
    def __init__(self, /, first_date, last_date, shape, file_path):
        self.file_path = file_path
        self.first_date = first_date
        self.last_date = last_date
        self.shape = shape
        self.offset = None

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
        json_path = file_path + ".json"
        if not os.path.exists(json_path) or (os.path.getmtime(json_path) <= os.path.getmtime(file_path)):
            array = np.load(file_path, **MMAP_KWARGS)
            cls.from_array(array, file_path=file_path).dump(json_path)
        return cls.from_json(json_path)

    @classmethod
    def from_json(cls, json_path: str):
        import json

        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(
            first_date=datetime.datetime.fromisoformat(data["first_date"]),
            last_date=datetime.datetime.fromisoformat(data["last_date"]),
            shape=tuple(data["shape"]),
            file_path=data["file_path"],
        )

    def dump(self, json_path: str):
        import json

        data = {
            "first_date": self.first_date.isoformat(),
            "last_date": self.last_date.isoformat(),
            "shape": self.shape,
            "file_path": self.file_path,
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)


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


def _unduplicate_worker(file_path: str, delete_file: bool):

    array = np.load(file_path, **MMAP_KWARGS)

    duplicates, unique_array = _unduplicate_rows(array)

    if duplicates:
        # LOG.warning(f"Removed {duplicates} duplicate rows from {file_path}")
        np.save(file_path + ".tmp", unique_array)  # Save to a temporary file first, numpy will add a .npy extension
        if delete_file:
            os.rename(file_path + ".tmp.npy", file_path)
        else:
            os.rename(file_path + ".tmp.npy", file_path + ".deduped.npy")
            file_path = file_path + ".deduped.npy"

    first_date = _date(unique_array, 0)
    last_date = _date(unique_array, -1)

    return duplicates, Chunk(first_date=first_date, last_date=last_date, shape=unique_array.shape, file_path=file_path)


def _deoverlap_worker(one, two, delete_files: bool):
    array_one = np.load(one.file_path, **MMAP_KWARGS)
    array_two = np.load(two.file_path, **MMAP_KWARGS)

    i = 0
    last = _date(array_one, -1)
    while i < array_two.shape[0] and last >= _date(array_two, i):
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
    if delete_files:
        os.rename(one.file_path + ".tmp.npy", one.file_path)
    else:
        os.rename(one.file_path + ".tmp.npy", one.file_path + ".deduped.npy")
        one.file_path = one.file_path + ".deduped.npy"

    np.save(two.file_path + ".tmp", end_array_two)
    if delete_files:
        os.rename(two.file_path + ".tmp.npy", two.file_path)
    else:
        os.rename(two.file_path + ".tmp.npy", two.file_path + ".deduped.npy")
        two.file_path = two.file_path + ".deduped.npy"

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


def _sort_and_chain_chunks(chunks):
    chunks = sorted(chunks, key=lambda x: x.first_date)
    offset = 0
    for chunk in chunks:
        chunk.offset = offset
        offset += chunk.shape[0]
    return chunks


def _test_only(work_dir: str):

    files = []
    for file in os.listdir(work_dir):
        if not file.endswith(".npy"):
            continue
        if ".tmp" in file or ".deduped" in file:
            continue

        files.append(os.path.join(work_dir, file))

    result = []
    for file in tqdm.tqdm(files, desc="Loading chunks"):
        result.append(Chunk.from_path(file))

    return _sort_and_chain_chunks(result)


def _find_duplicate_and_overlapping_dates(work_dir: str, delete_files, max_workers: int = None):

    import os

    chunks = {}
    total_duplicates = 0

    if max_workers is None:
        max_workers = max(int(os.cpu_count() * 0.7), 1)

    with Executor(max_workers=max_workers) as executor:

        tasks = []
        for file in os.listdir(work_dir):
            if not file.endswith(".npy"):
                continue
            if ".tmp" in file or ".deduped" in file:
                continue

            full_path = os.path.join(work_dir, file)
            tasks.append(executor.submit(_unduplicate_worker, full_path, delete_files))

        LOG.info("Checking duplicates")
        with tqdm.tqdm(total=len(tasks), desc="Checking duplicates", unit="file") as pbar:
            for future in as_completed(tasks):
                duplicates, chunk = future.result()
                total_duplicates += duplicates
                chunks[chunk.file_path] = chunk
                pbar.update(1)
                pbar.set_postfix({"duplicates": total_duplicates})

        LOG.info("Done")

        while True:

            tasks = []
            prev = None
            for chunk in sorted(chunks.values(), key=lambda x: x.first_date):
                if prev is None:
                    prev = chunk
                    continue

                if chunk.first_date <= prev.last_date:
                    tasks.append(executor.submit(_deoverlap_worker, prev, chunk, delete_files))
                    del chunks[prev.file_path]
                    del chunks[chunk.file_path]
                    prev = None
                else:
                    prev = chunk

            if not tasks:
                break

            LOG.info("Checking overlaps")
            with tqdm.tqdm(total=len(tasks), desc="Checking overlaps", unit="pair") as pbar:
                for future in as_completed(tasks):
                    duplicates, updates = future.result()
                    total_duplicates += duplicates
                    chunks.update({update.file_path: update for update in updates})
                    pbar.update(1)
                    pbar.set_postfix({"duplicates": total_duplicates})

    LOG.info(f"Total duplicates removed: {total_duplicates}")
    return _sort_and_chain_chunks(list(chunks.values()))


def finalise_tabular_dataset(*, store, work_dir: str, delete_files: bool, max_workers: int = None):

    if False:
        chunks = _test_only(work_dir)
    else:
        chunks = _find_duplicate_and_overlapping_dates(work_dir, max_workers=max_workers, delete_files=delete_files)

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

    with ChunksCache(store["data"]) as data:

        def _load_chunk(chunk):
            array = np.load(chunk.file_path)
            return (chunk, array)

        with ThreadPoolExecutor(max_workers=2) as executor:

            # Load 2 chunks ahead of time, so we have some sort of double buffering
            # while writing to Zarr

            tasks = []
            i = 0
            for i in range(len(chunks)):
                tasks.append(executor.submit(_load_chunk, chunks[i]))
                if i >= 2:
                    break

            i = len(tasks)

            with tqdm.tqdm(total=len(data), desc="Writing to Zarr", unit="row") as pbar:
                while tasks:
                    chunk, array = tasks.pop(0).result()
                    data[chunk.offset : chunk.offset + chunk.shape[0], :] = array
                    pbar.update(chunk.shape[0])
                    if i < len(chunks):
                        tasks.append(executor.submit(_load_chunk, chunks[i]))
                        i += 1


if __name__ == "__main__":
    import sys

    import zarr

    logging.basicConfig(level=logging.INFO)
    work_dir = sys.argv[2]
    store = zarr.open(sys.argv[1], mode="w")
    finalise_tabular_dataset(store=store, work_dir=work_dir, delete_files=False)
