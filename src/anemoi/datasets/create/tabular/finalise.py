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
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any
from typing import Optional

import numpy as np
import tqdm
import zarr

from anemoi.datasets.buffering import ReadAheadWriteBehindBuffer
from anemoi.datasets.create.statistics import StatisticsCollector
from anemoi.datasets.date_indexing import create_date_indexing

LOG = logging.getLogger(__name__)


class Fragment:
    """Represents a fragment of tabular data with associated date range and shape information.

    Parameters
    ----------
    first_date : datetime.datetime
        The first date in the fragment.
    last_date : datetime.datetime
        The last date in the fragment.
    shape : tuple of int
        The shape of the fragment array.
    file_path : str
        Path to the file containing the fragment data.
    """

    def __init__(
        self,
        /,
        first_date: datetime.datetime,
        last_date: datetime.datetime,
        shape: tuple[int, ...],
        file_path: str,
    ) -> None:
        """Initialise a Fragment instance.

        Parameters
        ----------
        first_date : datetime.datetime
            The first date in the fragment.
        last_date : datetime.datetime
            The last date in the fragment.
        shape : tuple of int
            The shape of the fragment array.
        file_path : str
            Path to the file containing the fragment data.
        """
        self.file_path: str = file_path
        self.first_date: datetime.datetime = first_date
        self.last_date: datetime.datetime = last_date
        self.shape: tuple[int, ...] = shape
        self.offset: int | None = None

    @classmethod
    def from_array(cls, array: np.ndarray, file_path: str) -> Optional["Fragment"]:
        """Create a Fragment instance from a numpy array.

        This method inspects the provided numpy array, which is expected to represent a fragment of tabular data
        with date information encoded in the first two columns. It extracts the first and last date from the array,
        determines the shape, and returns a new Fragment instance. If the array is empty, None is returned.

        Parameters
        ----------
        array : numpy.ndarray
            The array containing the fragment data. The first two columns should encode date information as (days, seconds).
        file_path : str
            Path to the file containing the fragment data.

        Returns
        -------
        Fragment or None
            The created Fragment instance, or None if the array is empty.
        """
        if len(array) == 0:
            return None
        # Dates are encoded as (days, seconds) in the first two columns
        first_date: datetime.datetime = _date(array, 0)
        last_date: datetime.datetime = _date(array, -1)
        shape: tuple[int, ...] = array.shape
        return cls(first_date=first_date, last_date=last_date, shape=shape, file_path=file_path)

    @classmethod
    def from_path(cls, file_path: str) -> "Fragment":
        """Create a Fragment instance from a file path.

        This method loads a numpy array from the specified file path (using memory mapping for efficiency),
        and then delegates to from_array to construct a Fragment instance. The file is expected to contain a 2D numpy array
        with date information in the first two columns. This is useful for reconstructing fragment metadata from disk.

        Parameters
        ----------
        file_path : str
            Path to the file containing the fragment data.

        Returns
        -------
        Fragment
            The created Fragment instance.
        """
        array: np.ndarray = np.load(file_path, mmap_mode="r")
        return cls.from_array(array, file_path=file_path)


def _deduplicate_rows(array: np.ndarray) -> np.ndarray:
    """Remove duplicate rows from a 2D numpy array, handling NaNs correctly.

    This function is designed to efficiently remove duplicate rows from a 2D numpy array, even when the array contains
    NaN values. Since numpy's unique function does not treat NaNs as equal, this function replaces NaNs with a sentinel
    value (infinity) before determining uniqueness. The function returns both the number of duplicate rows removed and
    the resulting array with only unique rows, preserving the original order as much as possible.

    Parameters
    ----------
    array : numpy.ndarray
        2D array from which to remove duplicate rows. NaNs are treated as equal for the purpose of deduplication.

    Returns
    -------
    numpy.ndarray
        Array with duplicates removed.
    """
    assert len(array.shape) == 2, f"Expected 2D array, got shape {array.shape}"
    # Remove duplicate rows. np.unique does not work well with NaNs, so we replace them with a sentinel value.

    b: np.ndarray = np.ascontiguousarray(array)
    # Replace NaNs with inf so np.unique can treat NaNs as equal
    b2: np.ndarray = np.nan_to_num(b, nan=np.inf)

    # Use a void dtype to treat each row as a single entity for uniqueness
    row_dtype: np.dtype = np.dtype((np.void, b2.dtype.itemsize * b2.shape[1]))
    _, idx = np.unique(b2.view(row_dtype), return_index=True)

    unique_array: np.ndarray = b[np.sort(idx)]

    return unique_array


def _date(array: np.ndarray, index: int) -> datetime.datetime:
    """Convert a row in the array to a datetime object.

    This function interprets the first two columns of the specified row as (days, seconds) since the Unix epoch,
    and returns the corresponding datetime.datetime object. This encoding is used throughout the fragment files to
    efficiently store date information as integers.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing date information, with the first two columns representing (days, seconds).
    index : int
        Index of the row to convert.

    Returns
    -------
    datetime.datetime
        The corresponding datetime object.
    """
    # Convert (days, seconds) to a datetime object
    return datetime.datetime.fromtimestamp(int(array[index][0]) * 86400 + int(array[index][1]))


def _deoverlap_worker(one: Fragment, two: Fragment, delete_files: bool) -> list[Fragment]:
    """Worker function to resolve overlapping date ranges between two fragments.

    This function is used to merge two fragments that have overlapping date ranges. It finds the point where the overlap ends,
    removes any duplicate row at the boundary, and then splits the data into two non-overlapping arrays. The function
    saves the new arrays to disk, either overwriting the originals or creating deduped versions, and returns the number
    of duplicates removed along with a list of new Fragment instances. This is essential for ensuring that the final dataset
    has strictly increasing, non-overlapping date ranges across all fragments.

    Parameters
    ----------
    one : Fragment
        The first fragment.
    two : Fragment
        The second fragment.
    delete_files : bool
        Whether to overwrite the original files or create deduped copies.

    Returns
    -------
    list of Fragment
        The resulting list of updated Fragment instances.
    """
    array_one: np.ndarray = np.load(one.file_path, mmap_mode="r")
    array_two: np.ndarray = np.load(two.file_path, mmap_mode="r")

    # Find the index in array_two where the overlap with array_one ends
    i: int = 0
    last: datetime.datetime = _date(array_one, -1)
    while i < array_two.shape[0] and last >= _date(array_two, i):
        i += 1

    assert i > 0

    beginning_array_two: np.ndarray = array_two[:i]

    # If the last row of array_one is identical to the first of array_two, skip it to avoid duplication
    if np.allclose(array_one[-1], beginning_array_two[0], equal_nan=True, rtol=0, atol=0):
        beginning_array_two = beginning_array_two[1:]

    # Merge the two arrays, removing overlap
    new_array_one: np.ndarray = np.vstack([array_one, beginning_array_two])
    end_array_two: np.ndarray = array_two[i:]

    # Save merged arrays, using temp files for atomicity
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

    result: list[Fragment] = []

    # Only keep non-empty arrays as fragments
    if new_array_one.size > 0:
        one = Fragment.from_path(one.file_path)
        assert one is not None
        result.append(one)
    else:
        os.unlink(one.file_path)

    if end_array_two.size > 0:
        two = Fragment.from_path(two.file_path)
        assert two is not None
        result.append(two)
    else:
        os.unlink(two.file_path)

    return result


def _sort_and_chain_fragments(fragments: list[Fragment]) -> list[Fragment]:
    """Sort fragments by first date and assign offsets for chaining.

    This function sorts a list of Fragment objects in ascending order of their first_date attribute, and then assigns
    a running offset to each fragment so that they can be concatenated into a single array. The offset is used to
    determine where each fragment's data should be written in the final output array. This is a preparatory step for
    efficiently writing the final dataset to disk in a single pass.

    Parameters
    ----------
    fragments : list of Fragment
        List of Fragment instances to sort and chain.

    Returns
    -------
    list of Fragment
        Sorted and offset-assigned fragments.
    """
    # Sort by first date and assign offsets for each fragment
    fragments = sorted(fragments, key=lambda x: x.first_date)
    offset: int = 0
    for fragment in fragments:
        fragment.offset = offset
        offset += fragment.shape[0]
    return fragments


def _list_files(work_dir: str) -> Generator[str, None, None]:
    """Yield file paths for .npy files in a working directory, excluding temporary and special files.

    This function scans the specified working directory and returns a list of all .npy files that are not temporary
    or special files (such as those used for date indices or intermediate deduplication). This is used to identify all
    candidate fragment files for further processing in the finalisation pipeline.

    Parameters
    ----------
    work_dir : str
        Directory to search for files.

    Yields
    -------
        Paths to valid .npy files.
    """

    for file in os.listdir(work_dir):
        # Exclude special and temporary files
        if file in ("dates.npy", "dates_ranges.npy"):
            continue

        if not file.endswith(".npy"):
            continue

        if ".tmp" in file or ".deduped" in file:
            continue

        yield os.path.join(work_dir, file)


def _read_fragment_worker(file_path: str) -> Fragment:
    return Fragment.from_path(file_path)


def _find_duplicate_and_overlapping_dates(
    work_dir: str, delete_files: bool, max_workers: int | None = None
) -> list[Fragment]:
    """Find and resolve duplicate and overlapping date ranges in fragment files.

    This function orchestrates the deduplication and deoverlapping of all fragment files in a working directory.
    It first removes duplicate rows from each fragment in parallel, then repeatedly detects and resolves overlaps
    between adjacent fragments until all fragments are strictly ordered and non-overlapping. The result is a list of
    Fragment objects that can be safely concatenated to form the final dataset. This is the main data cleaning step
    in the finalisation pipeline, ensuring data integrity and temporal consistency.

    Parameters
    ----------
    work_dir : str
        Directory containing fragment files.
    delete_files : bool
        Whether to overwrite original files or create deduped copies.
    max_workers : int, optional
        Maximum number of parallel workers to use. If None, uses all available CPUs.

    Returns
    -------
    list of Fragment
        List of deduplicated and deoverlapped Fragment instances.
    """
    import os

    fragments: dict[str, Fragment] = {}

    if max_workers is None:
        # For some reason using too many workers causes hangs in ProcessPoolExecutor
        max_workers = max(os.cpu_count() // 2, 1)

    LOG.info(f"Using {max_workers} workers for deduplication and deoverlapping")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Read all fragments in parallel

        tasks: list[Any] = []
        for file in _list_files(work_dir):
            tasks.append(executor.submit(_read_fragment_worker, file))

        LOG.info("Loading fragments")
        with tqdm.tqdm(total=len(tasks), desc="Loading fragments", unit="file") as pbar:
            for future in as_completed(tasks):
                fragment = future.result()
                if fragment is not None:
                    fragments[fragment.file_path] = fragment
                pbar.update(1)

        LOG.info("Checking overlaps")

        # Iteratively resolve overlaps until none remain
        while True:

            tasks = []
            prev: Fragment | None = None
            for fragment in sorted(fragments.values(), key=lambda x: x.first_date):
                if prev is None:
                    prev = fragment
                    continue

                if fragment.first_date <= prev.last_date:
                    # Overlap detected, resolve in parallel
                    tasks.append(executor.submit(_deoverlap_worker, prev, fragment, delete_files))
                    del fragments[prev.file_path]
                    del fragments[fragment.file_path]
                    prev = None
                else:
                    prev = fragment

            if not tasks:
                LOG.info("No more overlaps detected")
                break

            with tqdm.tqdm(total=len(tasks), desc="Checking overlaps", unit="pair") as pbar:
                for future in as_completed(tasks):
                    updates = future.result()
                    fragments.update({update.file_path: update for update in updates})
                    pbar.update(1)

        # There is a bug in ProcessPoolExecutor that hangs if the number of tasks sent is smaller
        # that the number of workers, so we send dummy tasks to avoid it.

        LOG.info("Done")

    return _sort_and_chain_fragments(list(fragments.values()))


def _duplicate_ranges(a: np.ndarray) -> list[tuple[int, int]]:
    """Find ranges of duplicate values in a sorted array.

    This function scans a sorted array and identifies contiguous runs of identical values, returning a list of
    (start, length) tuples for each such run. This is used to efficiently record the locations and extents of
    duplicate dates in the final dataset, which can then be indexed for fast lookup or further analysis.

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

    start = time.time()

    # True where value differs from previous => boundaries
    boundaries: np.ndarray = np.r_[True, a[1:] != a[:-1], True]

    # Find indices where value changes (start of new group)
    idx: np.ndarray = np.flatnonzero(boundaries)

    # Each (start, end) pair defines a run of identical values
    ranges: list[tuple[int, int]] = list(zip(idx[:-1], idx[1:]))

    LOG.info(f"Found {len(ranges):,} duplicate ranges out of {len(a):,} dates in {time.time()-start:.2f} seconds")

    return [(s, e - s) for s, e in ranges]


def _statistics_collector_worker(
    statistic_collector: StatisticsCollector,
    array: np.ndarray,
    dates: np.ndarray,
) -> None:
    """Worker function to collect statistics for a fragment of data.

    This function is designed to be executed in a separate thread or process. It takes a StatisticsCollector instance,
    an offset, a data array, and corresponding dates, and invokes the collect method of the StatisticsCollector.
    This allows for concurrent computation of statistics while the main thread handles I/O operations, improving
    overall performance during the finalisation process.

    Parameters
    ----------
    statistic_collector : StatisticsCollector
        The statistics collector instance to use.
    array : numpy.ndarray
        The data array for which to collect statistics.
    dates : numpy.ndarray
        The corresponding dates for the data array.
    """

    dates = dates.astype("datetime64[s]")
    statistic_collector.collect(array, dates)


def finalise_tabular_dataset(
    *,
    store: Any,
    work_dir: str,
    recipe: Any,
    variables_names: list[str],
    date_indexing: dict | str,
    delete_files: bool,
    max_workers: int | None = None,
) -> StatisticsCollector:
    """Finalise a tabular dataset by deduplicating, deoverlapping, and writing to a Zarr store.

    This is the main entry point for the tabular dataset finalisation process. It orchestrates the entire pipeline:
    - Deduplicates and deoverlaps all fragment files in the working directory.
    - Computes the final shape and fragmenting for the output dataset.
    - Writes the cleaned data to a Zarr store, using efficient fragmented I/O.
    - Extracts and records all duplicate date ranges for fast lookup.
    - Optionally deletes all intermediate files to save disk space.

    The result is a single, clean, and efficiently indexed tabular dataset ready for downstream analysis or distribution.

    """
    fragments: list[Fragment] = _find_duplicate_and_overlapping_dates(
        work_dir, max_workers=max_workers, delete_files=delete_files
    )

    assert fragments, "No data found to finalise"
    shape: tuple[int, int] = fragments[0].shape
    assert all(fragment.shape[1] == shape[1] for fragment in fragments), "Inconsistent number of columns in fragments"
    shape = (sum(fragment.shape[0] for fragment in fragments), fragments[0].shape[1])

    LOG.info(f"First fragment: {fragments[0].first_date} to {fragments[0].last_date}")
    LOG.info(f"Last fragment: {fragments[-1].first_date} to {fragments[-1].last_date}")

    collector = StatisticsCollector(
        variables_names=variables_names,
        filter=recipe.statistics.statistics_filter(
            [
                np.datetime64(fragments[0].first_date, "s"),
                np.datetime64(fragments[-1].last_date, "s"),
            ]
        ),
    )

    if "data" in store:
        del store["data"]

    # Choose fragment size to target ~64MB per fragment for efficient I/O
    row_size: int = shape[1] * np.dtype(np.float32).itemsize
    target_size: int = 64 * 1024 * 1024
    fragment_size: int = max(1, round(target_size / row_size))

    chunking: tuple[int, int] = (min(fragment_size, shape[0]), shape[1])
    LOG.info(f"Final dataset shape: {shape}, chunking: {chunking}")
    LOG.info(
        f"Number of rows: {shape[0]:,}, rows per chunk: {chunking[0]:,}, total chunks: {(shape[0] + chunking[0] - 1) // chunking[0]:,}"
    )

    store.create_dataset(
        "data",
        shape=shape,
        chunks=chunking,
        dtype=np.float32,
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    )

    all_dates_path: str = os.path.join(work_dir, "dates.npy")
    all_dates: np.ndarray = np.memmap(all_dates_path, dtype=np.int64, mode="w+", shape=(shape[0],))

    with ReadAheadWriteBehindBuffer(store["data"]) as data:

        def _load_fragment(fragment: Fragment) -> tuple[Fragment, np.ndarray]:
            # Remove deduped files after loading to save disk space
            array: np.ndarray = np.load(fragment.file_path)
            if "deduped" in fragment.file_path:
                os.unlink(fragment.file_path)
            return (fragment, array)

        with ThreadPoolExecutor(max_workers=2) as read_ahead, ThreadPoolExecutor(max_workers=1) as compute_statistics:

            # Double buffering: keep two fragments loaded ahead for performance
            tasks: list[Any] = []
            i: int = 0
            for i in range(len(fragments)):
                tasks.append(read_ahead.submit(_load_fragment, fragments[i]))
                if i >= 2:
                    break

            i = len(tasks)
            stats = None

            with tqdm.tqdm(total=len(data), desc="Writing to Zarr", unit="row") as pbar:
                while tasks:
                    fragment, array = tasks.pop(0).result()
                    data[fragment.offset : fragment.offset + fragment.shape[0], :] = array
                    dates = array[:, 0].astype(np.int64) * 86400 + array[:, 1].astype(np.int64)

                    # Wait for previous statistics computation to complete
                    if stats is not None:
                        stats.result()
                    stats = compute_statistics.submit(_statistics_collector_worker, collector, array, dates)

                    # Dates are encoded as (days, seconds) in columns 0 and 1
                    all_dates[fragment.offset : fragment.offset + fragment.shape[0]] = dates

                    pbar.update(fragment.shape[0])

                    if delete_files:
                        os.unlink(fragment.file_path)

                    # Pre-load next fragment
                    if i < len(fragments):
                        tasks.append(read_ahead.submit(_load_fragment, fragments[i]))
                        i += 1

            # Ensure last statistics computation is complete
            if stats is not None:
                stats.result()

    all_dates.flush()
    LOG.info(f"Dates written to {all_dates_path}")

    index = create_date_indexing(date_indexing, store)

    LOG.info("Compute duplicate date ranges")
    # Assume dates are sorted for efficient duplicate range finding
    ranges: list[tuple[int, int]] = _duplicate_ranges(all_dates)
    LOG.info(f"Found {len(ranges):,} duplicate date ranges")

    dates_ranges_path: str = os.path.join(work_dir, "dates_ranges.npy")
    dates_ranges: np.ndarray = np.memmap(dates_ranges_path, dtype=np.int64, mode="w+", shape=(len(ranges), 3))
    for i, (start, length) in enumerate(tqdm.tqdm(ranges, desc="Writing dates", unit="dates")):
        dates_ranges[i, :] = (all_dates[start], start, length)
    dates_ranges.flush()

    start = time.time()
    LOG.info("Bulking load duplicate date ranges into index")
    index.bulk_load(dates_ranges)
    LOG.info(f"Duplicate date ranges written to index in {time.time() - start:.2f} seconds")

    if delete_files:
        os.unlink(dates_ranges_path)
        os.unlink(all_dates_path)

    # Set the format attribute to indicate this is a tabular dataset
    store.attrs.update({"format": "tabular"})
    store.attrs.update({"date_indexing": index.name})

    return collector


if __name__ == "__main__":
    import sys

    import zarr

    logging.basicConfig(level=logging.INFO)
    work_dir = sys.argv[2]
    store = zarr.open(sys.argv[1], mode="w")
    collector = StatisticsCollector()

    finalise_tabular_dataset(
        store=store,
        work_dir=work_dir,
        statistic_collector=collector,
        delete_files=False,
    )

    with open(os.path.basename(sys.argv[1] + ".done"), "w") as f:
        pass

    for name in ("mean", "minimum", "maximum", "stdev"):
        store.create_dataset(
            name,
            data=collector.statistics()[name],
            shape=collector.statistics()[name].shape,
            dtype=collector.statistics()[name].dtype,
            overwrite=True,
        )
