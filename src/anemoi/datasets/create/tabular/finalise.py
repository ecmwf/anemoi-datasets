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
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from functools import cached_property
from typing import Any
from typing import Optional

import numpy as np
import pandas as pd
import psutil
import tqdm
import zarr

from anemoi.datasets.buffering import ReadAheadBuffer
from anemoi.datasets.buffering import WriteBehindBuffer
from anemoi.datasets.create.statistics import StatisticsCollector
from anemoi.datasets.date_indexing import create_date_indexing
from anemoi.datasets.epochs import epoch_to_date
from anemoi.datasets.memory import available_memory

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

        assert (
            self.first_date <= self.last_date
        ), f"Fragment {file_path} has invalid date range: {self.first_date} to {self.last_date}"

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

    @cached_property
    def size(self) -> int:
        """Return the size of the fragment in bytes."""
        return os.path.getsize(self.file_path)

    def __repr__(self) -> str:
        return f"Fragment({self.first_date} to {self.last_date}, shape={self.shape}, file={os.path.basename(self.file_path)})"


def _deduplicate_rows(array: np.ndarray) -> np.ndarray:
    """Removes duplicate rows from a NumPy array.
    This function converts the input NumPy array to a pandas DataFrame,
    removes any duplicate rows, and returns the deduplicated data as a NumPy array
    with the same dtype as the input.
    Parameters
    ----------
    array : np.ndarray
        The input NumPy array from which duplicate rows will be removed.
    Returns
    -------
    np.ndarray
        A NumPy array with duplicate rows removed, preserving the original dtype.
    """

    df = pd.DataFrame(array)
    deduped_df = df.drop_duplicates()
    return deduped_df.to_numpy(dtype=array.dtype)


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
    return epoch_to_date(int(array[index][0]) * 86400 + int(array[index][1]))


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

    try:

        # Not mmapping gives up a 3x speedup here, but may lead to memkills on large files
        extra = dict(mmap_mode="r")
        extra = dict()  # --- IGNORE ---

        array_one = np.load(one.file_path, **extra)
        array_two = np.load(two.file_path, **extra)

        # Find the range where both arrays have identical first two columns (date columns)
        # Concatenate to find the overlap region around the junction
        concatenated = np.vstack([array_one, array_two])
        junction_index = len(array_one)

        # Find the range where dates are identical around the junction
        # Look backwards from junction to find where duplicates start
        start_idx = junction_index - 1

        while start_idx > 0 and np.array_equal(concatenated[start_idx, :2], concatenated[junction_index, :2]):
            start_idx -= 1
        if not np.array_equal(concatenated[start_idx, :2], concatenated[junction_index, :2]):
            start_idx += 1

        # Look forwards from junction to find where duplicates end
        end_idx = junction_index
        junction_date = concatenated[junction_index, :2]
        while end_idx < len(concatenated) and np.array_equal(concatenated[end_idx, :2], junction_date):
            end_idx += 1

        LOG.debug(f"Found overlapping date range  {start_idx}:{end_idx} around junction at {junction_index}")

        # Check for duplicate rows in the overlap region
        overlap_region = concatenated[start_idx:end_idx]
        num_duplicates = 0
        deduped_region = None
        if overlap_region.shape[0] > 1:
            deduped_region = _deduplicate_rows(overlap_region)
            num_duplicates = overlap_region.shape[0] - deduped_region.shape[0]

        if num_duplicates == 0:
            return [one, two]

        LOG.info(f"Duplicate rows found in overlapping fragments {one} and {two} ({num_duplicates=:,}); deoverlapping")

        result = []

        base = f"{one.file_path}-{time.time()}"

        first_region = overlap_region[:start_idx]
        if len(first_region) > 0:
            np.save(base + ".tmp", first_region)
            os.rename(base + ".tmp.npy", base + ".deduped.one.npy")
            result.append(Fragment.from_path(base + ".deduped.one.npy"))

        second_region = overlap_region[end_idx:]
        if len(second_region) > 0:
            np.save(base + ".tmp", second_region)
            os.rename(base + ".tmp.npy", base + ".deduped.two.npy")
            result.append(Fragment.from_path(base + ".deduped.two.npy"))

        np.save(base + ".tmp", deduped_region)
        os.rename(base + ".tmp.npy", base + ".deduped.overlap.npy")
        overlap_fragment = Fragment.from_path(base + ".deduped.overlap.npy")

        assert overlap_fragment is not None
        result.append(overlap_fragment)

        return result

    except Exception as e:
        LOG.error(f"Error deoverlapping fragments {one.file_path} and {two.file_path}: {e}")
        LOG.exception("Error in deoverlap_worker")
        raise


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
    try:
        return Fragment.from_path(file_path)
    except Exception:
        LOG.exception(f"Error reading fragment from {file_path}")
        try:
            array: np.ndarray = np.load(file_path, mmap_mode="r")
            LOG.error(f"Array shape: {array.shape}, dtype: {array.dtype}")
            LOG.error(f"First row: {array[0] if len(array) > 0 else 'N/A'}")
            LOG.error(f"Last row: {array[-1] if len(array) > 0 else 'N/A'}")
        except Exception:
            LOG.exception(f"Error loading array from {file_path}")
        raise


def _find_duplicate_and_overlapping_dates(
    work_dir: str,
    delete_files: bool,
    max_fragment_size: int,
    max_workers: int | None = None,
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
    max_fragment_size : int
        Maximum size of each fragment file in bytes. This is used to estimate memory requirements for parallel
    max_workers : int, optional
        Maximum number of parallel workers to use. If None, uses all available CPUs.

    Returns
    -------
    list of Fragment
        List of deduplicated and deoverlapped Fragment instances.
    """
    import os

    fragments: dict[str, Fragment] = {}

    memory = available_memory()
    # TODO: read value from recipe

    cpus = os.cpu_count() or 1
    if "SLURM_CPUS_ON_NODE" in os.environ:
        cpus = min(cpus, int(os.environ["SLURM_CPUS_ON_NODE"]))

    LOG.info(f"Available memory: {memory / (1024**3):.2f} GB")
    LOG.info(f"Available CPUs: {cpus}")

    memory *= 0.8  # Use only 80% of available memory

    max_fragment_size = 256 * 1024 * 1024  # 256 MB

    # Each pairs of deoverlapping fragments requires loading both into memory.
    # Then double it for safety
    me = psutil.Process(os.getpid())

    # Assume each worker needs 4x max_fragment_size + current memory

    my_memory = me.memory_full_info().rss
    LOG.info(f"Current process memory usage: {my_memory / (1024**3):.2f} GB")

    estimated_needed_memory = 6 * max_fragment_size + my_memory
    estimated_needed_memory *= 1.2  # Safety margin

    estimated_max_workers = int(memory / estimated_needed_memory)

    LOG.info(f"Estimated max workers based on memory: {estimated_max_workers}")

    if max_workers is None:
        max_workers = min(max(cpus - 1, 1), estimated_max_workers)
    else:
        LOG.info(
            f"User requested max_workers={max_workers}, estimated max_workers={estimated_max_workers} based on memory"
        )

    LOG.info(f"Using {max_workers} workers for deduplication and deoverlapping")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        # Read all fragments in parallel

        tasks: list[Any] = []

        for file in _list_files(work_dir):
            tasks.append(executor.submit(_read_fragment_worker, file))

        LOG.info("Loading fragments")
        now = time.time()
        with tqdm.tqdm(total=len(tasks), desc="Loading fragments", unit="file") as pbar:
            for future in as_completed(tasks):
                fragment = future.result()
                if fragment is not None:
                    fragments[fragment.file_path] = fragment
                pbar.update(1)

        LOG.info(f"Loaded {len(fragments):,} fragments in {time.time()-now:.2f} seconds")

        now = time.time()
        LOG.info("Checking overlaps")

        # Iteratively resolve overlaps until none remain
        seen = set()
        while True:

            tasks = []
            prev: Fragment | None = None
            for fragment in sorted(fragments.values(), key=lambda x: x.first_date):
                if prev is None:
                    prev = fragment
                    continue

                if fragment.first_date <= prev.last_date and (prev.file_path, fragment.file_path) not in seen:
                    # Overlap detected, resolve in parallel
                    tasks.append(executor.submit(_deoverlap_worker, prev, fragment, delete_files))
                    seen.add((prev.file_path, fragment.file_path))
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

        LOG.info("Overlap checking complete")
        LOG.info(f"Resolved overlaps in {time.time()-now:.2f} seconds")

    return _sort_and_chain_fragments(list(fragments.values()))


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

    now = time.time()

    dates = dates.astype("datetime64[s]")
    statistic_collector.collect(array, dates)

    return time.time() - now


class _DuplicateRangeBuilder:  # (value, start_index, length)
    def __init__(self, length: int, path: str) -> None:
        # Length of the input array to process, used for pre-allocating the output file
        # Resulting array will be smaller but we don't know the exact size until we process it, so we use the input length as an upper bound
        self.total_size = length
        self.path = path

        # Create an empty file (sparse if supported) to hold the output ranges
        # We don't uss np.memmap(mode='w')  because that may trigger an OOM if the file is large,
        # even if it's sparse.Becaue Numpy will access all pages to initialize them,
        # which can cause the OS to allocate physical memory for the entire file.
        # By using open() and truncate(), we create a sparse file without triggering OOM.

        self.row_size = 3 * np.dtype(np.int64).itemsize  # Each row has 3 int64 values: (value, start_index, length)
        bytes_needed = self.total_size * self.row_size

        with open(self.path, "wb") as f:
            f.truncate(bytes_needed)

        # Mmap the file for writing the output ranges
        self.dates_ranges = np.memmap(self.path, dtype=np.int64, mode="r+", shape=(length, 3))

        self.current_run_start = 0  # To keep track of where to write the next range in the output
        self.current_val = None

        self.chunk = 0

        self.start_idx = 0
        self.range_idx = 0
        self.current_start_idx = 0

    def _add_range(self, chunk: np.ndarray, data_slice: slice) -> None:

        if self.current_val is None:
            self.current_val = chunk[0]

        now = time.time()

        # Identify Change Points (where the NEXT run starts)
        diffs = chunk[:-1] != chunk[1:]
        internal_changes = np.flatnonzero(diffs) + 1 + self.start_idx

        # Create a unified list of global indices where a change occurs
        change_points = internal_changes

        # If the start of this chunk differs from what we were tracking,
        # the boundary itself (start_idx) is a change point.
        if chunk[0] != self.current_val:
            change_points = np.concatenate(([self.start_idx], change_points))

        # Remove duplicates and sort.
        # This prevents the 'current_start_idx == end_point' scenario that creates 0-length entries.
        change_points = np.unique(change_points)

        # Process the Timeline
        for end_point in change_points:
            # Strictly: end_point is where the NEW value begins.
            # Therefore, the OLD value ends at end_point.
            length = end_point - self.current_start_idx

            self.dates_ranges[self.range_idx] = (self.current_val, self.current_start_idx, length)
            self.range_idx += 1

            # Update state
            self.current_start_idx = end_point
            self.current_val = chunk[end_point - self.start_idx]

        self.start_idx += len(chunk)
        return time.time() - now

    def array(self) -> np.ndarray:

        final_len = self.total_size - self.current_start_idx
        if final_len > 0:
            self.dates_ranges[self.range_idx] = (self.current_val, self.current_start_idx, final_len)
            self.range_idx += 1

        del self.dates_ranges  # Close the mapping to allow truncation
        with open(self.path, "ab") as f:
            f.truncate(self.range_idx * self.row_size)

        return np.memmap(self.path, dtype=np.int64, mode="r", shape=(self.range_idx, 3))


def _build_duplicate_ranges_worker(builder, dates: np.ndarray, data_slice: slice) -> None:
    try:
        return builder._add_range(dates, data_slice)
    except Exception:
        LOG.exception("Error processing chunk for duplicate range building")
        raise


def finalise_tabular_dataset(
    *,
    store: Any,
    work_dir: str,
    recipe: Any,
    variables_names: list[str],
    date_indexing: dict | str,
    delete_files: bool,
    offset: int = 4,
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
        work_dir,
        max_fragment_size=recipe.build.max_fragment_size,
        max_workers=recipe.build.max_workers,
        delete_files=delete_files,
    )

    assert fragments, "No data found to finalise"
    shape: tuple[int, int] = fragments[0].shape
    assert all(fragment.shape[1] == shape[1] for fragment in fragments), "Inconsistent number of columns in fragments"
    shape = (sum(fragment.shape[0] for fragment in fragments), fragments[0].shape[1])

    LOG.info(f"First fragment: {fragments[0].first_date} to {fragments[0].last_date}")
    LOG.info(f"Last fragment: {fragments[-1].first_date} to {fragments[-1].last_date}")

    dates = []
    date = fragments[0].first_date
    last = fragments[-1].last_date
    while date <= last:
        dates.append(date)
        date += datetime.timedelta(days=1)

    dates = np.array(dates, dtype="datetime64[s]")

    collector = StatisticsCollector(
        variables_names=variables_names,
        filter=recipe.statistics.statistics_filter(dates),
    )

    dates_ranges_path = os.path.join(work_dir, "dates_ranges.npy")

    date_range_builer = _DuplicateRangeBuilder(length=shape[0], path=dates_ranges_path)

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

    with WriteBehindBuffer(store["data"]) as data:

        def _load_fragment(fragment: Fragment) -> tuple[Fragment, np.ndarray]:
            # Remove deduped files after loading to save disk space
            array: np.ndarray = np.load(fragment.file_path)
            if "deduped" in fragment.file_path:
                os.unlink(fragment.file_path)
            return (fragment, array)

        with (
            ThreadPoolExecutor(max_workers=2) as read_ahead,
            ThreadPoolExecutor(max_workers=1) as compute_statistics,
            ThreadPoolExecutor(max_workers=1) as build_duplicate_ranges,
        ):

            # Double buffering: keep two fragments loaded ahead for performance
            tasks: list[Any] = []
            i: int = 0
            for i in range(len(fragments)):
                tasks.append(read_ahead.submit(_load_fragment, fragments[i]))
                if i >= 2:
                    break

            i = len(tasks)
            stats = None
            stat_time = 0
            data_time = 0
            date_time = 0

            build = None
            build_time = 0
            with tqdm.tqdm(total=len(data), desc="Writing to Zarr", unit="row") as pbar:
                while tasks:
                    fragment, array = tasks.pop(0).result()

                    now = time.time()
                    data[fragment.offset : fragment.offset + fragment.shape[0], :] = array
                    data_time += time.time() - now

                    # Dates are encoded as (days, seconds) in columns 0 and 1
                    now = time.time()
                    dates = array[:, 0].astype(np.int64) * 86400 + array[:, 1].astype(np.int64)
                    date_time += time.time() - now

                    # Wait for previous statistics computation to complete
                    if stats is not None:
                        stat_time += stats.result()

                    stats = compute_statistics.submit(_statistics_collector_worker, collector, array[:, offset:], dates)

                    # Wait for previous duplicate range building to complete
                    if build is not None:
                        build_time += build.result()

                    build = build_duplicate_ranges.submit(
                        _build_duplicate_ranges_worker,
                        date_range_builer,
                        dates,
                        slice(fragment.offset, fragment.offset + fragment.shape[0]),
                    )

                    pbar.update(fragment.shape[0])

                    if delete_files:
                        os.unlink(fragment.file_path)

                    # Pre-load next fragment
                    if i < len(fragments):
                        tasks.append(read_ahead.submit(_load_fragment, fragments[i]))
                        i += 1

            # Ensure last statistics computation is complete
            if stats is not None:
                stat_time += stats.result()

            if build is not None:
                build_time += build.result()

            LOG.info(f"Statistics computed in {stat_time:.2f} seconds ({len(data)/stat_time:.2f} rows/second).")
            LOG.info(f"Data written in {data_time:.2f} seconds ({len(data)/data_time:.2f} rows/second).")
            LOG.info(f"Dates written in {date_time:.2f} seconds ({len(data)/date_time:.2f} rows/second).")
            LOG.info(
                f"Duplicate date ranges built in {build_time:.2f} seconds ({len(data)/build_time:.2f} rows/second)."
            )

    dates_ranges = date_range_builer.array()

    LOG.info(f"Duplicate date ranges written to {dates_ranges_path} with {len(dates_ranges):,} ranges")

    ############################# Validation of date ranges (can be removed in production for performance) #############################
    LOG.info("Validating date ranges")
    with ReadAheadBuffer(store["data"]) as data:
        # Check that the number of ranges found matches the count we got during building
        offset = 0
        last_date = None
        for i, (date, start, length) in enumerate(tqdm.tqdm(dates_ranges, desc="Validating date ranges", unit="row")):
            assert (
                length > 0
            ), f"Found non-positive range for date {date} starting at index {start} ({(date, start, length)}) [{i=}]"
            assert start == offset, f"Found non-contiguous range starting at {start}, expected {offset} [{i=}]"
            assert last_date is None or date > last_date, f"Found non-increasing date {date} after {last_date} [{i=}]"

            chunk = data[start : start + length, :]
            date_column = chunk[:, 0].astype(np.int64) * 86400 + chunk[:, 1].astype(np.int64)
            # Check that column 0 (days since epoch) of data matches the current date
            assert np.all(
                date_column == date
            ), f"Mismatch between date range {date} and data column 0 at rows {start}:{start+length} ({date_column=}) [{i=}]"

            offset += length
            last_date = date

        assert offset == shape[0], f"Total length of ranges {offset} does not match total number of rows {shape[0]}"
    ############################# End of validation #############################

    index = create_date_indexing(date_indexing, store)

    start = time.time()
    LOG.info("Bulking load duplicate date ranges into index")
    index.bulk_load(dates_ranges)
    LOG.info(f"Duplicate date ranges written to index in {time.time() - start:.2f} seconds")

    del dates_ranges

    if delete_files:
        os.unlink(dates_ranges_path)

    # Set the format attribute to indicate this is a tabular dataset
    store.attrs.update({"layout": "tabular"})
    store.attrs.update({"date_indexing": index.name})

    return collector


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    _find_duplicate_and_overlapping_dates(sys.argv[1], delete_files=False, max_workers=None)
