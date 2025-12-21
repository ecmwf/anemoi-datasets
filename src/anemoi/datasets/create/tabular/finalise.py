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
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import tqdm

from anemoi.datasets.tabular.btree import ZarrBTree
from anemoi.datasets.tabular.caching import ChunksCache

LOG = logging.getLogger(__name__)


class Chunk:
    """Represents a chunk of tabular data with associated date range and shape information.

    Parameters
    ----------
    first_date : datetime.datetime
        The first date in the chunk.
    last_date : datetime.datetime
        The last date in the chunk.
    shape : tuple of int
        The shape of the chunk array.
    file_path : str
        Path to the file containing the chunk data.
    """

    def __init__(
        self,
        /,
        first_date: datetime.datetime,
        last_date: datetime.datetime,
        shape: Tuple[int, ...],
        file_path: str,
    ) -> None:
        """Initialise a Chunk instance.

        Parameters
        ----------
        first_date : datetime.datetime
            The first date in the chunk.
        last_date : datetime.datetime
            The last date in the chunk.
        shape : tuple of int
            The shape of the chunk array.
        file_path : str
            Path to the file containing the chunk data.
        """
        self.file_path: str = file_path
        self.first_date: datetime.datetime = first_date
        self.last_date: datetime.datetime = last_date
        self.shape: Tuple[int, ...] = shape
        self.offset: Optional[int] = None

    @classmethod
    def from_array(cls, array: np.ndarray, file_path: str) -> Optional["Chunk"]:
        """Create a Chunk instance from a numpy array.

        Parameters
        ----------
        array : numpy.ndarray
            The array containing the chunk data.
        file_path : str
            Path to the file containing the chunk data.

        Returns
        -------
        Chunk or None
            The created Chunk instance, or None if the array is empty.
        """
        if len(array) == 0:
            return None
        first_date: datetime.datetime = _date(array, 0)
        last_date: datetime.datetime = _date(array, -1)
        shape: Tuple[int, ...] = array.shape
        return cls(first_date=first_date, last_date=last_date, shape=shape, file_path=file_path)

    @classmethod
    def from_path(cls, file_path: str) -> "Chunk":
        """Create a Chunk instance from a file path.

        Parameters
        ----------
        file_path : str
            Path to the file containing the chunk data.

        Returns
        -------
        Chunk
            The created Chunk instance.
        """

        array: np.ndarray = np.load(file_path, mmap_mode="r")
        return cls.from_array(array, file_path=file_path)


def _unduplicate_rows(array: np.ndarray) -> Tuple[int, np.ndarray]:
    """Remove duplicate rows from a 2D numpy array, handling NaNs correctly.

    Parameters
    ----------
    array : numpy.ndarray
        2D array from which to remove duplicate rows.

    Returns
    -------
    int
        Number of duplicate rows removed.
    numpy.ndarray
        Array with duplicates removed.
    """
    assert len(array.shape) == 2, f"Expected 2D array, got shape {array.shape}"
    # Remove duplicate rows. np.unique does not work well with NaNs, so we replace them with a sentinel value.

    b: np.ndarray = np.ascontiguousarray(array)
    b2: np.ndarray = np.nan_to_num(b, nan=np.inf)

    row_dtype: np.dtype = np.dtype((np.void, b2.dtype.itemsize * b2.shape[1]))
    _, idx = np.unique(b2.view(row_dtype), return_index=True)

    unique_array: np.ndarray = b[np.sort(idx)]

    duplicates: int = len(array) - len(unique_array)
    return duplicates, unique_array


def _date(array: np.ndarray, index: int) -> datetime.datetime:
    """Convert a row in the array to a datetime object.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing date information.
    index : int
        Index of the row to convert.

    Returns
    -------
    datetime.datetime
        The corresponding datetime object.
    """
    return datetime.datetime.fromtimestamp(int(array[index][0]) * 86400 + int(array[index][1]))


def _unduplicate_worker(file_path: str, delete_file: bool) -> Tuple[int, Chunk]:
    """Worker function to remove duplicate rows from a file and update the file as needed.

    Parameters
    ----------
    file_path : str
        Path to the file to process.
    delete_file : bool
        Whether to overwrite the original file or create a deduped copy.

    Returns
    -------
    int
        Number of duplicate rows removed.
    Chunk
        The resulting Chunk instance.
    """
    array: np.ndarray = np.load(file_path, mmap_mode="r")

    duplicates, unique_array = _unduplicate_rows(array)

    if duplicates:
        # LOG.warning(f"Removed {duplicates} duplicate rows from {file_path}")
        np.save(file_path + ".tmp", unique_array)  # Save to a temporary file first, numpy will add a .npy extension
        if delete_file:
            os.rename(file_path + ".tmp.npy", file_path)
        else:
            os.rename(file_path + ".tmp.npy", file_path + ".deduped.npy")
            file_path = file_path + ".deduped.npy"

    first_date: datetime.datetime = _date(unique_array, 0)
    last_date: datetime.datetime = _date(unique_array, -1)

    return duplicates, Chunk(
        first_date=first_date,
        last_date=last_date,
        shape=unique_array.shape,
        file_path=file_path,
    )


def _deoverlap_worker(one: Chunk, two: Chunk, delete_files: bool) -> Tuple[int, List[Chunk]]:
    """Worker function to resolve overlapping date ranges between two chunks.

    Parameters
    ----------
    one : Chunk
        The first chunk.
    two : Chunk
        The second chunk.
    delete_files : bool
        Whether to overwrite the original files or create deduped copies.

    Returns
    -------
    int
        Number of duplicate rows removed.
    list of Chunk
        The resulting list of updated Chunk instances.
    """
    array_one: np.ndarray = np.load(one.file_path, mmap_mode="r")
    array_two: np.ndarray = np.load(two.file_path, mmap_mode="r")

    i: int = 0
    last: datetime.datetime = _date(array_one, -1)
    while i < array_two.shape[0] and last >= _date(array_two, i):
        i += 1

    assert i > 0

    duplicates: int = 0

    beginning_array_two: np.ndarray = array_two[:i]

    if np.allclose(array_one[-1], beginning_array_two[0], equal_nan=True, rtol=0, atol=0):
        duplicates = 1
        beginning_array_two = beginning_array_two[1:]

    new_array_one: np.ndarray = np.vstack([array_one, beginning_array_two])
    end_array_two: np.ndarray = array_two[i:]

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

    result: List[Chunk] = []

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


def _sort_and_chain_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Sort chunks by first date and assign offsets for chaining.

    Parameters
    ----------
    chunks : list of Chunk
        List of Chunk instances to sort and chain.

    Returns
    -------
    list of Chunk
        Sorted and offset-assigned chunks.
    """
    chunks = sorted(chunks, key=lambda x: x.first_date)
    offset: int = 0
    for chunk in chunks:
        chunk.offset = offset
        offset += chunk.shape[0]
    return chunks


def _list_files(work_dir: str) -> Any:
    """Yield file paths for .npy files in a working directory, excluding temporary and special files.

    Parameters
    ----------
    work_dir : str
        Directory to search for files.

    Yields
    ------
    str
        Path to a valid .npy file.
    """
    for file in os.listdir(work_dir):
        if file in ("dates.npy", "dates_ranges.npy"):
            continue

        if not file.endswith(".npy"):
            continue

        if ".tmp" in file or ".deduped" in file:
            continue

        yield os.path.join(work_dir, file)


def _test_only(work_dir: str) -> List[Chunk]:
    """Load all chunks from a working directory for testing purposes.

    Parameters
    ----------
    work_dir : str
        Directory containing chunk files.

    Returns
    -------
    list of Chunk
        List of loaded Chunk instances.
    """
    files: List[str] = list(_list_files(work_dir))

    result: List[Chunk] = []
    for file in tqdm.tqdm(files, desc="Loading chunks"):
        result.append(Chunk.from_path(file))

    return _sort_and_chain_chunks(result)


def _find_duplicate_and_overlapping_dates(
    work_dir: str, delete_files: bool, max_workers: Optional[int] = None
) -> List[Chunk]:
    """Find and resolve duplicate and overlapping date ranges in chunk files.

    Parameters
    ----------
    work_dir : str
        Directory containing chunk files.
    delete_files : bool
        Whether to overwrite original files or create deduped copies.
    max_workers : int, optional
        Maximum number of parallel workers to use.

    Returns
    -------
    list of Chunk
        List of deduplicated and deoverlapped Chunk instances.
    """
    import os

    chunks: Dict[str, Chunk] = {}
    total_duplicates: int = 0

    if max_workers is None:
        max_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        tasks: List[Any] = []
        for file in _list_files(work_dir):
            tasks.append(executor.submit(_unduplicate_worker, file, delete_files))

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
            prev: Optional[Chunk] = None
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


def _duplicate_ranges(a: np.ndarray) -> List[Tuple[int, int]]:
    """Find ranges of duplicate values in a sorted array.

    Parameters
    ----------
    a : numpy.ndarray
        Sorted array of values.

    Returns
    -------
    list of tuple of int
        List of (start, length) tuples for each duplicate range.
    """
    if a.size == 0:
        return []

    # True where value differs from previous => boundaries
    boundaries: np.ndarray = np.r_[True, a[1:] != a[:-1], True]

    # Indices where boundaries occur
    idx: np.ndarray = np.flatnonzero(boundaries)

    # Pair successive boundaries → (start,end) for each group
    ranges: List[Tuple[int, int]] = list(zip(idx[:-1], idx[1:]))

    return [(s, e - s) for s, e in ranges]


def finalise_tabular_dataset(
    *, store: Any, work_dir: str, delete_files: bool, max_workers: Optional[int] = None
) -> None:
    """Finalise a tabular dataset by deduplicating, deoverlapping, and writing to a Zarr store.

    Parameters
    ----------
    store : Any
        Zarr store or similar object to write the final dataset to.
    work_dir : str
        Directory containing chunk files.
    delete_files : bool
        Whether to delete temporary files after processing.
    max_workers : int, optional
        Maximum number of parallel workers to use.
    """
    chunks: List[Chunk] = _find_duplicate_and_overlapping_dates(
        work_dir, max_workers=max_workers, delete_files=delete_files
    )

    assert chunks, "No data found to finalise"
    shape: Tuple[int, int] = chunks[0].shape
    assert all(chunk.shape[1] == shape[1] for chunk in chunks), "Inconsistent number of columns in chunks"
    shape = (sum(chunk.shape[0] for chunk in chunks), chunks[0].shape[1])

    if "data" in store:
        del store["data"]

    row_size: int = shape[1] * np.dtype(np.float32).itemsize
    target_size: int = 64 * 1024 * 1024  # Target chunk size of 64 MB
    chunk_size: int = max(1, round(target_size / row_size))

    chunking: Tuple[int, int] = (min(chunk_size, shape[0]), shape[1])
    LOG.info(f"Final dataset shape: {shape}, chunking: {chunking}")
    store.create_dataset(
        "data",
        shape=shape,
        chunks=chunking,
        dtype=np.float32,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    )

    dates_path: str = os.path.join(work_dir, "dates.npy")
    dates: np.ndarray = np.memmap(dates_path, dtype=np.int64, mode="w+", shape=(shape[0],))

    with ChunksCache(store["data"]) as data:

        def _load_chunk(chunk: Chunk) -> Tuple[Chunk, np.ndarray]:
            array: np.ndarray = np.load(chunk.file_path)
            if "deduped" in chunk.file_path:
                os.unlink(chunk.file_path)
            return (chunk, array)

        with ThreadPoolExecutor(max_workers=2) as executor:

            # Load 2 chunks ahead of time, so we have some sort of double buffering
            # while writing to Zarr

            tasks: List[Any] = []
            i: int = 0
            for i in range(len(chunks)):
                tasks.append(executor.submit(_load_chunk, chunks[i]))
                if i >= 2:
                    break

            i = len(tasks)

            with tqdm.tqdm(total=len(data), desc="Writing to Zarr", unit="row") as pbar:
                while tasks:
                    chunk, array = tasks.pop(0).result()
                    data[chunk.offset : chunk.offset + chunk.shape[0], :] = array

                    # Extract dates
                    dates[chunk.offset : chunk.offset + chunk.shape[0]] = array[:, 0].astype(np.int64) * 86400 + array[
                        :, 1
                    ].astype(np.int64)

                    pbar.update(chunk.shape[0])

                    if delete_files:
                        os.unlink(chunk.file_path)

                    # Pre-load next chunk
                    if i < len(chunks):
                        tasks.append(executor.submit(_load_chunk, chunks[i]))
                        i += 1

    dates.flush()
    LOG.info(f"Dates written to {dates_path}")
    LOG.info("Compute duplicate date ranges")
    # Assume dates are sorted
    ranges: List[Tuple[int, int]] = _duplicate_ranges(dates)

    dates_ranges_path: str = os.path.join(work_dir, "dates_ranges.npy")
    dates_ranges: np.ndarray = np.memmap(dates_ranges_path, dtype=np.int64, mode="w+", shape=(len(ranges), 3))
    for i, (start, length) in tqdm.tqdm(enumerate(ranges), desc="Writing duplicate date ranges", unit="range"):
        dates_ranges[i, :] = (dates[start], start, length)
    dates_ranges.flush()

    LOG.info(f"Found {len(dates_ranges)} duplicate date ranges")
    btree: ZarrBTree = ZarrBTree(store)
    btree.bulk_load(dates_ranges)
    LOG.info("Duplicate date ranges written to B-Tree index")

    if delete_files:
        LOG.info("Deleting temporary files")
        os.unlink(dates_path)
        os.unlink(dates_ranges_path)


if __name__ == "__main__":
    import sys

    import zarr

    logging.basicConfig(level=logging.INFO)
    work_dir = sys.argv[2]
    store = zarr.open(sys.argv[1], mode="w")
    finalise_tabular_dataset(store=store, work_dir=work_dir, delete_files=False)
    with open(os.path.basename(sys.argv[1] + ".done"), "w") as f:
        pass
