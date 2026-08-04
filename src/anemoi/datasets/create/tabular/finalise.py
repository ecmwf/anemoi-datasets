# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import glob
import hashlib
import logging
import os
import pickle
import socket
import threading
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

from anemoi.datasets.compat import blosc_compressor
from anemoi.datasets.create.statistics import StatisticsCollector
from anemoi.datasets.date_indexing import create_date_indexing
from anemoi.datasets.epochs import array_to_epoch
from anemoi.datasets.epochs import epoch_to_date
from anemoi.datasets.memory import available_memory

LOG = logging.getLogger(__name__)


LOG_LOCK = threading.Lock()


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
        self._file_path: str = file_path
        self._first_date: datetime.datetime = first_date
        self._last_date: datetime.datetime = last_date
        self._shape: tuple[int, ...] = shape
        self._offset: int | None = None

        assert (
            self.first_date <= self.last_date
        ), f"Fragment {file_path} has invalid date range: {self.first_date} to {self.last_date}"

    @property
    def file_path(self) -> str:
        """Get the file path of the fragment."""
        return self._file_path

    @property
    def first_date(self) -> datetime.datetime:
        """Get the first date of the fragment."""
        return self._first_date

    @property
    def last_date(self) -> datetime.datetime:
        """Get the last date of the fragment."""
        return self._last_date

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the fragment array."""
        return self._shape

    @property
    def offset(self) -> int | None:
        """Get the offset of the fragment in the final dataset."""
        return self._offset

    @offset.setter
    def offset(self, value: int) -> None:
        """Set the offset of the fragment in the final dataset.

        Parameters
        ----------
        value : int
            The offset value to set.
        """
        assert (
            self._offset is None
        ), f"Offset for fragment {self.file_path} is already set to {self._offset}, cannot overwrite with {value}"
        self._offset = value

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
        array.flags.writeable = False

        return cls.from_array(array, file_path=file_path)

    @cached_property
    def size(self) -> int:
        """Return the size of the fragment in bytes."""
        return os.path.getsize(self.file_path)

    def __repr__(self) -> str:
        return f"[{self.first_date}; {self.last_date}; {self.shape}; {os.path.basename(self.file_path)}]"


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


def _path(dirname: str, array: np.ndarray, short_hash: str) -> str:
    first_date = _date(array, 0)
    last_date = _date(array, -1)
    return os.path.join(dirname, f"{first_date.isoformat()}-{last_date.isoformat()}-{short_hash}")


_DEDUP_JOURNAL = "dedup_journal.log"


def _fsync(path: str) -> None:
    """Flush a file (or directory) to stable storage so a rename/write survives a power loss."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _dedup_journal_path(work_dir: str) -> str:
    return os.path.join(work_dir, _DEDUP_JOURNAL)


def _replay_dedup_journal(work_dir: str) -> None:
    """Reclaim the superseded input files recorded by a previous (crashed) dedup run.

    Each line is the path of a fragment whose de-overlapped outputs were durably written before
    the crash (that is guaranteed by the write ordering in the dedup loop), so it is dead weight
    and safe to delete. Idempotent: inputs already gone are skipped.
    """
    path = _dedup_journal_path(work_dir)
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            victim = line.strip()
            if victim:
                _unlink_if_exists(victim)


def _deoverlap_worker(one: Fragment, two: Fragment) -> list[Fragment]:
    """Worker function to resolve overlapping date ranges between two fragments.

    This function merges two fragments that have overlapping date ranges. It finds the point where the overlap ends,
    removes any duplicate rows, and splits the data into two non-overlapping arrays, written as **new** ``.deduped``
    files. The outputs are ``fsync``-ed (and the directory too) before returning, so that once the caller sees the
    result the outputs are durable. The worker does **not** delete the two inputs: the caller does that, after
    journalling, so that an input is only ever removed once its data is safely captured in the durable outputs.

    Parameters
    ----------
    one : Fragment
        The first fragment.
    two : Fragment
        The second fragment.

    Returns
    -------
    list of Fragment
        The resulting list of new Fragment instances.
    """

    try:
        # Not mmapping gives up a 3x speedup here, but may lead to memkills on large files
        extra = dict(mmap_mode="r")
        extra = dict()  # --- IGNORE ---

        array_one = np.load(one.file_path, **extra)
        array_two = np.load(two.file_path, **extra)

        concat = np.vstack([array_one, array_two])
        del array_one
        del array_two

        concat = _deduplicate_rows(concat)

        _, counts = np.unique(concat[:, :2], axis=0, return_counts=True)
        sum_ = np.sum(counts)
        cumsum = np.cumsum(counts)
        half_point = np.searchsorted(cumsum, sum_ // 2)
        # assert False, (half_point, np.sum(counts[:half_point]), np.sum(counts[half_point:]))

        split_point = np.sum(counts[:half_point])

        result = []

        pid = os.getpid()
        hostname = socket.gethostname()
        timestamp = str(time.time())
        hash_input = f"{pid}{hostname}{timestamp}".encode()
        dirname = os.path.dirname(one.file_path)
        short_hash = hashlib.sha1(hash_input).hexdigest()[:7]

        # Write each half as a new `.deduped` file; the originals `one`/`two` are left for the
        # caller to delete once these outputs are durable. Empty halves simply drop out.
        outputs: list[str] = []
        for region, suffix in ((concat[:split_point], "deduped.1"), (concat[split_point:], "deduped.2")):
            if len(region) == 0:
                continue
            path = _path(dirname, region, short_hash)
            np.save(path + ".tmp", region)
            out = f"{path}.{suffix}.npy"
            os.rename(path + ".tmp.npy", out)
            outputs.append(out)
            result.append(Fragment.from_path(out))

        # Make the outputs durable (data + directory entry) before the caller deletes the inputs,
        # so a power loss in that window can never lose data.
        for out in outputs:
            _fsync(out)
        _fsync(dirname)

        with LOG_LOCK:
            LOG.info(f"Deoverlapping fragments\n    {one}\n    {two}")
            LOG.info("\n -> ".join([""] + [repr(r) for r in result]))

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
    previous_date = None
    for fragment in fragments:
        if previous_date is not None and fragment.first_date <= previous_date:
            raise ValueError(
                f"Fragment {fragment.file_path} has first date {fragment.first_date} which is before last date {previous_date} of previous fragment. This may indicate an overlap that was not resolved."
            )
        fragment.offset = offset
        offset += fragment.shape[0]
        previous_date = fragment.last_date

    return fragments


def _list_files(work_dir: str) -> Generator[str, None, None]:
    """Yield the live fragment ``.npy`` files in a working directory.

    Returns every ``.npy`` file that is a candidate fragment, excluding only special files and the ``.tmp`` files of
    an in-flight write. ``.deduped`` files **are** included: because the dedup loop deletes each input as soon as its
    de-overlapped outputs are durable, on a re-run the surviving ``.deduped`` files are the live representation of the
    data and must be picked up. (Partially written ``.tmp`` files are excluded because they may be truncated.)

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

        if ".tmp" in file:
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
    max_fragment_size: int = 256 * 1024 * 1024,  # 256 MB
    max_workers: int | None = None,
) -> list[Fragment]:
    """Find and resolve duplicate and overlapping date ranges in fragment files.

    This function orchestrates the deduplication and deoverlapping of all fragment files in a working directory.
    It first removes duplicate rows from each fragment in parallel, then repeatedly detects and resolves overlaps
    between adjacent fragments until all fragments are strictly ordered and non-overlapping. The result is a list of
    Fragment objects that can be safely concatenated to form the final dataset. This is the main data cleaning step
    in the finalisation pipeline, ensuring data integrity and temporal consistency.

    Overlap resolution never modifies a fragment in place: each pair is written to new ``.deduped`` files and its two
    inputs are deleted only once those outputs are durable, so disk is reclaimed as the work proceeds (a fragment may
    be de-overlapped repeatedly, so keeping every generation could blow up disk use). A journal of deleted inputs
    (fsync'd before each delete) lets an interrupted run reclaim stragglers on restart, and because an input is only
    ever removed after its data is safely in the durable outputs, an interruption never loses data.

    Parameters
    ----------
    work_dir : str
        Directory containing fragment files.
    max_fragment_size : int
        Maximum size of each fragment file in bytes. This is used to estimate memory requirements for parallel
        processing.
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

    # Each pairs of deoverlapping fragments requires loading both into memory.
    # Then double it for safety
    me = psutil.Process(os.getpid())

    # Assume each worker needs 4x max_fragment_size + current memory

    my_memory = me.memory_full_info().rss
    LOG.info(f"Current process memory usage: {my_memory / (1024**3):.2f} GB")

    estimated_needed_memory = 6 * max_fragment_size + my_memory
    estimated_needed_memory *= 1.2  # Safety margin

    estimated_max_workers = max(1, int(memory / estimated_needed_memory))

    LOG.info(f"Estimated max workers based on memory: {estimated_max_workers}")

    if max_workers is None:
        max_workers = min(max(cpus - 1, 1), estimated_max_workers)
    else:
        LOG.info(
            f"User requested max_workers={max_workers}, estimated max_workers={estimated_max_workers} based on memory"
        )

    LOG.info(f"Using {max_workers} workers for deduplication and deoverlapping")

    # A previous run may have been killed after a pair's outputs were made durable but before its
    # inputs were deleted; reclaim those stragglers before we scan, so we don't carry duplicates.
    _replay_dedup_journal(work_dir)

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

        LOG.info(f"Loaded {len(fragments):,} fragments in {time.time() - now:.2f} seconds")

        now = time.time()
        LOG.info("Checking overlaps")

        # Reclaim disk as we go: once a pair's de-overlapped outputs are durable, its two inputs are
        # dead weight. A fragment may be re-deduplicated many times (the outputs of one pair can
        # overlap another and be split again), so keeping every generation could blow disk up by
        # orders of magnitude on a large build. We record each resolved pair's inputs in a journal
        # (fsync'd) and delete them immediately; a crashed run reclaims any stragglers on restart via
        # _replay_dedup_journal. Deleting an input is safe because the worker fsync'd the outputs
        # that contain its data before returning.
        journal = open(_dedup_journal_path(work_dir), "w")
        try:
            # Iteratively resolve overlaps until none remain
            seen = set()
            while True:
                tasks = []
                task_inputs: dict[Any, tuple[str, str]] = {}
                prev: Fragment | None = None
                for fragment in sorted(fragments.values(), key=lambda x: x.first_date):
                    if prev is None:
                        prev = fragment
                        continue

                    if fragment.first_date <= prev.last_date and (prev.file_path, fragment.file_path) not in seen:
                        # Overlap detected, resolve in parallel
                        future = executor.submit(_deoverlap_worker, prev, fragment)
                        task_inputs[future] = (prev.file_path, fragment.file_path)
                        tasks.append(future)
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

                        # The pair's data is now held entirely by the durable outputs, so journal the
                        # two inputs (fsync) and delete them to free their disk right away.
                        if updates:
                            inputs = task_inputs[future]
                            journal.write("".join(f"{p}\n" for p in inputs))
                            journal.flush()
                            os.fsync(journal.fileno())
                            for p in inputs:
                                _unlink_if_exists(p)
                        pbar.update(1)
        finally:
            journal.close()

        # Dedup finished: every surviving fragment is a live `.deduped` file, so the journal is
        # obsolete and can be removed.
        _unlink_if_exists(_dedup_journal_path(work_dir))

        # There is a bug in ProcessPoolExecutor that hangs if the number of tasks sent is smaller
        # that the number of workers, so we send dummy tasks to avoid it.

        LOG.info("Overlap checking complete")
        LOG.info(f"Resolved overlaps in {time.time() - now:.2f} seconds")

    return _sort_and_chain_fragments(list(fragments.values()))


def _statistics_collector_worker(
    statistic_collector: StatisticsCollector,
    array: np.ndarray,
    epochs: np.ndarray,
) -> None:
    """Worker function to collect statistics for a fragment of data.

    This function is designed to be executed in a separate thread or process. It takes a StatisticsCollector instance,
    an offset, a data array, and corresponding epochs, and invokes the collect method of the StatisticsCollector.
    This allows for concurrent computation of statistics while the main thread handles I/O operations, improving
    overall performance during the finalisation process.

    Parameters
    ----------
    statistic_collector : StatisticsCollector
        The statistics collector instance to use.
    array : numpy.ndarray
        The data array for which to collect statistics.
    epochs : numpy.ndarray
        The corresponding epochs for the data array.
    """

    now = time.time()

    dates = epochs.astype("datetime64[s]")
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
        self.last_date = None
        self.range_idx = 0

    def _add_range(self, dates: np.ndarray, data_slice: slice, fragment: Fragment) -> None:

        now = time.time()
        assert len(dates) > 0

        first_date = epoch_to_date(dates[0])
        last_date = epoch_to_date(dates[-1])

        assert (
            first_date == fragment.first_date
        ), f"First date {first_date} does not match fragment first date {fragment.first_date}  ({type(first_date)=}) ({type(fragment.first_date)=} {first_date-fragment.first_date=})"

        assert (
            last_date == fragment.last_date
        ), f"Last date {last_date} does not match fragment last date {fragment.last_date} {fragment.last_date-last_date=})"

        if self.last_date is not None and dates[0] <= self.last_date:
            raise ValueError(
                f"Dates are not strictly increasing: {dates[0]} <= last date {self.last_date} ({first_date} <= {epoch_to_date(self.last_date)})"
            )

        self.last_date = dates[-1]

        assert np.all(dates[:-1] <= dates[1:]), "Dates must be sorted in ascending order"

        # assert False, dates

        unique_dates, counts = np.unique(dates, return_counts=True)

        # Add cumulative sum of counts, starting with 0 for the first row
        offsets = np.concatenate(([0], np.cumsum(counts)[:-1])) + data_slice.start
        assert offsets[-1] + counts[-1] == data_slice.stop, (offsets[-1] + counts[-1], data_slice)
        result = np.column_stack((unique_dates, offsets, counts))
        size = len(result)

        self.dates_ranges[self.range_idx : self.range_idx + size, :] = result

        self.range_idx += size

        return time.time() - now

    def array(self) -> np.ndarray:

        del self.dates_ranges  # Close the mapping to allow truncation
        with open(self.path, "ab") as f:
            f.truncate(self.range_idx * self.row_size)

        return np.memmap(self.path, dtype=np.int64, mode="r", shape=(self.range_idx, 3))


def _build_duplicate_ranges_worker(builder, dates: np.ndarray, data_slice: slice, fragment: Fragment) -> None:
    try:
        return builder._add_range(dates, data_slice, fragment)
    except Exception:
        LOG.exception("Error processing chunk for duplicate range building")
        raise


def _load_fragment_worker(fragment: Fragment) -> tuple[Fragment, np.ndarray]:
    try:
        now = time.time()
        # Fragment files are only deleted once the whole part is committed (see
        # load_tabular_dataset), never here, so that a re-run can reload them.
        array: np.ndarray = np.load(fragment.file_path)

        first = _date(array, 0)
        last = _date(array, -1)

        assert (
            first == fragment.first_date
        ), f"First date {first} does not match fragment first date {fragment.first_date} for file {fragment.file_path}"
        assert (
            last == fragment.last_date
        ), f"Last date {last} does not match fragment last date {fragment.last_date} for file {fragment.file_path}"

        return (fragment, array, time.time() - now)
    except Exception:
        LOG.exception("Error loading fragment")
        raise


_MANIFEST_NAME = "finalise_manifest.pkl"


def _manifest_path(work_dir: str) -> str:
    return os.path.join(work_dir, _MANIFEST_NAME)


def _write_manifest(work_dir: str, manifest: dict) -> None:
    # Write then rename so the manifest only ever appears complete: its presence is
    # the marker that the `prepare` stage finished.
    path = _manifest_path(work_dir)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(manifest, f)
    os.replace(path + ".tmp", path)


def _read_manifest(work_dir: str) -> dict:
    with open(_manifest_path(work_dir), "rb") as f:
        return pickle.load(f)


# Every finalise stage can be killed at any point (e.g. by SLURM) and re-run. A small
# journal of ``<name>.done`` marker files in the work dir records which units of work have
# been committed, so a re-run skips them. Markers are written last, after the work they
# guard is safely on disk, and always via write-then-rename so they never appear partial.
# The overall "finalise finished" signal lives in the store (`_FINALISE_COMPLETE_ATTR`), so
# that re-running any stage after the work dir has been cleaned up is a no-op.

_FINALISE_COMPLETE_ATTR = "_tabular_finalise_complete"


def _finalise_complete(store: Any) -> bool:
    return bool(store.attrs.get(_FINALISE_COMPLETE_ATTR, False))


def _marker_path(work_dir: str, name: str) -> str:
    return os.path.join(work_dir, f"{name}.done")


def _is_marked_done(work_dir: str, name: str) -> bool:
    return os.path.exists(_marker_path(work_dir, name))


def _mark_done(work_dir: str, name: str) -> None:
    path = _marker_path(work_dir, name)
    with open(path + ".tmp", "w") as f:
        f.write("done\n")
    os.replace(path + ".tmp", path)


def _unlink_if_exists(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _delete_fragment_files(fragments: list["Fragment"]) -> None:
    for fragment in fragments:
        _unlink_if_exists(fragment.file_path)


def _reclaim_unreferenced_fragments(work_dir: str, manifest: dict) -> None:
    """Delete fragment files that the committed manifest no longer references.

    Overlap resolution is non-destructive, so once ``prepare`` has written the manifest the work
    dir holds both the surviving fragments (untouched originals and the final ``.deduped`` files,
    all listed in the manifest) *and* the superseded ones (originals that were de-overlapped and
    intermediate ``.deduped`` / ``.tmp`` files). The superseded files are pure dead weight — their
    data is fully contained in the manifest's fragments — so they are removed here to reclaim disk
    before the load stage, which matters when a de-overlap can transiently double disk usage on a
    multi-terabyte build. Idempotent: it only ever deletes ``.npy`` files absent from the manifest,
    so a re-run (e.g. after an interrupted sweep) simply finishes the job.
    """
    referenced = {os.path.realpath(fragment[0]) for fragment in manifest["fragments"]}
    for name in os.listdir(work_dir):
        if not name.endswith(".npy") or name in ("dates.npy", "dates_ranges.npy"):
            continue
        path = os.path.join(work_dir, name)
        if os.path.realpath(path) in referenced:
            continue
        _unlink_if_exists(path)


def _daily_epochs(first_date: datetime.datetime, last_date: datetime.datetime) -> np.ndarray:
    """Rebuild the daily date axis used to derive the statistics date range."""
    epochs = []
    date = first_date
    while date <= last_date:
        epochs.append(date)
        date += datetime.timedelta(days=1)
    return np.array(epochs, dtype="datetime64[s]")


def _fragment_part_bounds(parts: str | list | None, total: int) -> tuple[int, int, int]:
    """Return ``(part_index, lo, hi)`` for the fragment slice ``[lo, hi)`` of a part.

    Mirrors :class:`~anemoi.datasets.create.parts.PartFilter`'s 1-based ``i/n``
    contiguous split. The bounds are derived from the global fragment count so
    that empty parts (``n`` larger than the number of fragments) still yield
    ``lo == hi`` at the correct boundary, keeping the per-part statistics
    contiguous for ``tidy``.
    """
    if isinstance(parts, list):
        if len(parts) != 1:
            raise ValueError(f"Invalid parts format: {parts}. Must be a single 'i/n'.")
        parts = parts[0]

    if not parts or parts in ("all", "*"):
        return 0, 0, total

    i, n = (int(x) for x in parts.split("/"))
    chunk_size = total / n
    lo = sum(1 for x in range(total) if x < (i - 1) * chunk_size)
    hi = sum(1 for x in range(total) if x < i * chunk_size)
    return i - 1, lo, hi


def _stream_fragments_into_zarr(
    dataset: Any,
    fragments: list[Fragment],
    collector: StatisticsCollector,
    date_range_builder: "_DuplicateRangeBuilder",
    offset: int,
) -> None:
    """Write a list of fragments into ``store["data"]`` at their offsets.

    Statistics collection and duplicate-date-range building run on background
    threads while the next fragment is read ahead, mirroring the original
    single-pass pipeline. Writes go straight to the zarr array (zarr handles the
    read-modify-write of partial boundary chunks). Fragment files are not deleted
    here; the caller removes them only once the whole part is committed, so that a
    re-run of an interrupted part still finds every fragment on disk.
    """
    data = dataset.store["data"]

    with (
        ThreadPoolExecutor(max_workers=2) as read_ahead,
        ThreadPoolExecutor(max_workers=1) as compute_statistics,
        ThreadPoolExecutor(max_workers=1) as build_duplicate_ranges,
    ):
        # Double buffering: keep two fragments loaded ahead for performance
        tasks: list[Any] = []
        for i in range(len(fragments)):
            tasks.append(read_ahead.submit(_load_fragment_worker, fragments[i]))
            if i >= 1:  # Keep two fragments max : i=0 and i=1, then stop
                break

        i = len(tasks)
        stats = None
        build = None
        previous_date = None

        total_rows = sum(f.shape[0] for f in fragments)
        with tqdm.tqdm(total=total_rows, desc="Writing to Zarr", unit="row") as pbar:
            while tasks:
                fragment, array, _ = tasks.pop(0).result()

                if previous_date is not None and fragment.first_date <= previous_date:
                    raise ValueError(
                        f"Fragment {fragment.file_path} has first date {fragment.first_date} which is before "
                        f"last date {previous_date} of previous fragment. This may indicate an unresolved overlap."
                    )
                previous_date = fragment.last_date

                data[fragment.offset : fragment.offset + fragment.shape[0], :] = array

                # Dates are encoded as (days, seconds) in columns 0 and 1
                epochs = array_to_epoch(array)

                # Wait for previous statistics computation to complete
                if stats is not None:
                    stats.result()
                stats = compute_statistics.submit(_statistics_collector_worker, collector, array[:, offset:], epochs)

                # Wait for previous duplicate range building to complete
                if build is not None:
                    build.result()
                build = build_duplicate_ranges.submit(
                    _build_duplicate_ranges_worker,
                    date_range_builder,
                    epochs,
                    slice(fragment.offset, fragment.offset + fragment.shape[0]),
                    fragment,
                )

                pbar.update(fragment.shape[0])

                # Pre-load next fragment
                if i < len(fragments):
                    tasks.append(read_ahead.submit(_load_fragment_worker, fragments[i]))
                    i += 1

            # Ensure the last statistics/range computations are complete
            if stats is not None:
                stats.result()
            if build is not None:
                build.result()


def prepare_tabular_dataset(
    *,
    dataset: Any,
    work_dir: str,
    recipe: Any,
    variables_names: list[str],
    delete_files: bool,
    offset: int = 4,
) -> None:
    """Prepare stage: deduplicate, compute the shape, create the empty zarr array, write the manifest.

    Deduplicates and deoverlaps all fragment files, computes and validates the final shape and
    chunking, creates the (empty) ``data`` array in the store, and writes a manifest to ``work_dir``
    listing every fragment and its offset in the final zarr. Once the manifest is committed, the
    superseded fragment files are removed (when ``delete_files``) to reclaim disk before the load
    stage. The surviving fragment files are left on disk for the load stage. This step must run
    exactly once before any ``load`` stage.

    Re-runnable: if the manifest already exists the stage only finishes reclaiming disk (it never
    re-deduplicates or wipes the partially loaded ``data`` array); an interrupted de-overlap loses
    no data because it never modifies the original fragments.
    """
    store = dataset.store

    if _finalise_complete(store):
        LOG.info("Tabular finalise already complete; prepare is a no-op.")
        return

    if os.path.exists(_manifest_path(work_dir)):
        LOG.info("Manifest already present; prepare has already run.")
        if delete_files:
            # A previous run may have been killed mid-reclaim; finish it (idempotent).
            _reclaim_unreferenced_fragments(work_dir, _read_manifest(work_dir))
        return

    fragments: list[Fragment] = _find_duplicate_and_overlapping_dates(
        work_dir,
        max_fragment_size=recipe.build.max_fragment_size,
        max_workers=recipe.build.max_workers,
    )

    assert fragments, "No data found to finalise"
    shape: tuple[int, int] = fragments[0].shape
    assert all(fragment.shape[1] == shape[1] for fragment in fragments), "Inconsistent number of columns in fragments"
    shape = (sum(fragment.shape[0] for fragment in fragments), fragments[0].shape[1])

    # Backstop for the resume path: fragments reused from already-done groups
    # are never revalidated against the metadata at load time, so confirm the
    # on-disk width still matches the recorded schema before writing.
    expected_cols = len(store.attrs["meta_variables"]) + len(variables_names)
    assert shape[1] == expected_cols, (
        f"Fragments have {shape[1]} columns but metadata declares {expected_cols} "
        f"({len(store.attrs['meta_variables'])} meta + {len(variables_names)} variables)"
    )

    LOG.info(f"First fragment: {fragments[0].first_date} to {fragments[0].last_date}")
    LOG.info(f"Last fragment : {fragments[-1].first_date} to {fragments[-1].last_date}")

    row_size: int = shape[1] * np.dtype(np.float32).itemsize
    rows_per_chunk: int = max(1, round(recipe.output.bytes_per_chunk / row_size))
    if recipe.output.rows_per_chunk is not None:
        rows_per_chunk = recipe.output.rows_per_chunk

    chunking: tuple[int, int] = (min(rows_per_chunk, shape[0]), shape[1])
    LOG.info(f"Final dataset shape: {shape}, chunking: {chunking}")
    LOG.info(
        f"Number of rows: {shape[0]:,}, rows per chunk: {chunking[0]:,}, "
        f"total chunks: {(shape[0] + chunking[0] - 1) // chunking[0]:,}"
    )

    if "data" in store:
        del store["data"]

    store.create_array(
        "data",
        shape=shape,
        chunks=chunking,
        dtype=np.float32,
        compressors=blosc_compressor(cname="zstd", clevel=3, shuffle=2),
    )

    manifest = {
        "shape": shape,
        "chunking": chunking,
        "offset": offset,
        "first_date": fragments[0].first_date,
        "last_date": fragments[-1].last_date,
        "fragments": [(f.file_path, f.first_date, f.last_date, tuple(f.shape), f.offset) for f in fragments],
    }
    _write_manifest(work_dir, manifest)

    # The manifest now durably references every surviving fragment, so the superseded originals
    # and de-overlap intermediates can be reclaimed to avoid carrying ~2x the data through load.
    if delete_files:
        _reclaim_unreferenced_fragments(work_dir, manifest)

    LOG.info(f"Prepared {len(fragments):,} fragments for a final shape of {shape}.")


def _part_ranges_path(work_dir: str, part_index: int) -> str:
    return os.path.join(work_dir, f"dates_ranges_{part_index:06d}.bin")


def load_tabular_dataset(
    *,
    dataset: Any,
    work_dir: str,
    recipe: Any,
    variables_names: list[str],
    parts: str | list | None,
    delete_files: bool,
) -> None:
    """Load stage: write one part of the fragments into the zarr array.

    Reads the manifest, selects this part's contiguous slice of fragments, writes them into the
    already-created ``data`` array, and emits a per-part partial :class:`StatisticsCollector`
    (pickled, mirroring the gridded creator) and a per-part duplicate-date-ranges file. Both are
    combined in the ``tidy`` stage. ``parts`` is ``None`` (all fragments) or ``"i/n"`` (1-based).

    Re-runnable: a ``load_<part>.done`` marker is written last, once the part's data,
    statistics and date ranges are all committed. A re-run of a completed part only finishes
    deleting its fragment files; a re-run of an interrupted part recomputes it from scratch
    (the fragment files are only deleted after the marker, so they are still present).
    """
    store = dataset.store

    if _finalise_complete(store):
        LOG.info("Tabular finalise already complete; load is a no-op.")
        return

    if not os.path.exists(_manifest_path(work_dir)):
        raise RuntimeError(f"No manifest in {work_dir!r}: run `anemoi-datasets finalise --prepare` before `--load`.")

    manifest = _read_manifest(work_dir)

    fragments: list[Fragment] = []
    for file_path, first_date, last_date, shape, off in manifest["fragments"]:
        fragment = Fragment(first_date=first_date, last_date=last_date, shape=shape, file_path=file_path)
        fragment.offset = off
        fragments.append(fragment)

    total = len(fragments)
    part_index, lo, hi = _fragment_part_bounds(parts, total)
    part_fragments = fragments[lo:hi]

    final_rows = manifest["shape"][0]
    start = fragments[lo].offset if lo < total else final_rows
    end = fragments[hi].offset if hi < total else final_rows

    marker = f"load_{part_index:06d}"
    if _is_marked_done(work_dir, marker):
        LOG.info(f"Part {part_index} already loaded; ensuring its fragment files are removed.")
        if delete_files:
            _delete_fragment_files(part_fragments)
        return

    LOG.info(f"Loading part {part_index} ({parts}): fragments [{lo}:{hi}] of {total}, rows [{start}:{end}).")

    offset = manifest["offset"]
    epochs = _daily_epochs(manifest["first_date"], manifest["last_date"])
    collector = StatisticsCollector(
        variables_names=variables_names,
        filter=recipe.statistics.statistics_filter(epochs),
    )

    ranges_path = _part_ranges_path(work_dir, part_index)
    if part_fragments:
        date_range_builder = _DuplicateRangeBuilder(length=end - start, path=ranges_path)
        _stream_fragments_into_zarr(dataset, part_fragments, collector, date_range_builder, offset)
        date_range_builder.array()  # close and truncate the ranges file to its used size
    else:
        # Empty part: still emit an (empty) ranges file so tidy sees every part.
        open(ranges_path, "wb").close()

    stats_path = os.path.join(work_dir, f"statistics_{part_index:06d}-{start:09d}-{end:09d}.pkl")
    collector.serialise(stats_path + ".tmp", group=part_index, start=start, end=end)
    os.replace(stats_path + ".tmp", stats_path)

    # Marker last: once it exists, the part's data, statistics and ranges are all committed.
    _mark_done(work_dir, marker)

    # Only now is it safe to free the fragment files. If we are killed mid-deletion and
    # re-run, the marker short-circuits to the deletion branch above.
    if delete_files:
        _delete_fragment_files(part_fragments)

    LOG.info(f"Loaded part {part_index}: wrote rows [{start}:{end}) and saved partial statistics.")


def tidy_tabular_dataset(
    *,
    dataset: Any,
    work_dir: str,
    recipe: Any,
    variables_names: list[str],
    date_indexing: dict | str,
    delete_files: bool,
) -> None:
    """Tidy stage: merge the partial statistics and date ranges, build the index, set attrs.

    Only the parts that carry a ``load_<part>.done`` marker are combined, so an incomplete
    load can never contribute a half-written statistics or ranges file. Merges every part's
    statistics pickle (via :meth:`StatisticsCollector.load_precomputed`), concatenates the
    per-part date-ranges files in part order, validates them, builds the date index, writes
    the statistics and metadata into the dataset, and finally marks the finalise complete in
    the store before removing the manifest and all temporary files.

    Re-runnable: if the store already carries the completion flag the stage is a no-op, so a
    re-run after (or during) the clean-up does nothing.
    """
    store = dataset.store

    if _finalise_complete(store):
        LOG.info("Tabular finalise already complete; tidy is a no-op.")
        return

    # Only consider parts whose load committed (marker present), in part order.
    done_markers = sorted(glob.glob(_marker_path(work_dir, "load_*")))
    part_indices = [int(os.path.basename(p)[len("load_") : -len(".done")]) for p in done_markers]
    assert part_indices, "No completed load parts found; run the load stage before tidy."

    stats_files: list[str] = []
    range_files: list[str] = []
    for idx in part_indices:
        matches = glob.glob(os.path.join(work_dir, f"statistics_{idx:06d}-*.pkl"))
        assert len(matches) == 1, f"Expected exactly one statistics file for part {idx}, found {matches}"
        stats_files.append(matches[0])
        range_files.append(_part_ranges_path(work_dir, idx))

    # Merge the per-part partial statistics (mirrors the gridded creator).
    collector = StatisticsCollector.load_precomputed(dataset, stats_files)

    # Concatenate the per-part duplicate-date-range files in part order.
    row_size = 3 * np.dtype(np.int64).itemsize
    part_arrays = []
    for path in range_files:
        n_rows = os.path.getsize(path) // row_size
        if n_rows:
            part_arrays.append(np.memmap(path, dtype=np.int64, mode="r", shape=(n_rows, 3)))

    assert part_arrays, "No date ranges found; run the load stage before tidy."
    dates_ranges = part_arrays[0] if len(part_arrays) == 1 else np.concatenate(part_arrays)

    LOG.info(f"Duplicate date ranges: {len(dates_ranges):,} ranges from {len(part_arrays)} part(s)")

    if recipe.build.validate_date_ranges:
        from .validate import validate_date_ranges

        validate_date_ranges(store["data"], dates_ranges)

    index = create_date_indexing(date_indexing, store)

    now = time.time()
    LOG.info("Bulking load duplicate date ranges into index")
    index.bulk_load(dates_ranges)
    LOG.info(f"Duplicate date ranges written to index in {time.time() - now:.2f} seconds")

    del dates_ranges
    del part_arrays

    # Write the statistics into the dataset.
    collector.add_to_dataset(dataset)

    # Set the format attributes and the date-indexing metadata for the catalogue.
    store.attrs.update({"layout": "tabular"})
    store.attrs.update({"date_indexing": index.name})

    LOG.info("Computing date indexing metadata for the dataset.")
    first, last = index.start_end_dates()
    dataset.update_metadata(
        index_start_date=first.isoformat(),
        index_end_date=last.isoformat(),
        index_length=index.length(),
    )

    # Mark complete *before* removing anything: after this point a re-run is a no-op, so
    # there is no window in which the temporary files are gone but the flag is unset.
    store.attrs.update({_FINALISE_COMPLETE_ATTR: True})

    if delete_files:
        for path in stats_files + range_files + done_markers:
            _unlink_if_exists(path)
        _unlink_if_exists(_manifest_path(work_dir))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    _find_duplicate_and_overlapping_dates(sys.argv[1], max_workers=None)
