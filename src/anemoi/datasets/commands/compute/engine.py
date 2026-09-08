# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Orchestration for the ``compute`` command.

Drives the standalone accumulators over one (or, for residuals, two) datasets with
a simple chunked loop. Adds the production features: NaN policy, time-based
checkpointing, resume, and optional process-level parallelism. The numerics live
entirely in :mod:`statistics`, :mod:`statistics_tendencies` and
:mod:`statistics_residuals`; this module only schedules and merges them.
"""

import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .interpolation import DEFAULT_GRID_METHOD
from .interpolation import Interpolator
from .interpolation import TargetGrid
from .output_dataset import ConstantsTracker
from .output_dataset import OutputDataset
from .output_dataset import derived_from
from .statistics import DEFAULT_CHUNK_SIZE
from .statistics import Accumulator
from .statistics import iter_chunks
from .statistics_residuals import _check_compatible
from .statistics_tendencies import TendencyAccumulator
from .statistics_tendencies import delta_to_steps

LOG = logging.getLogger(__name__)

CHECKPOINT_VERSION = 3  # bumped when 'Collectors' gained the constants tracker
CHECKPOINT_INTERVAL = 60.0  # seconds
LIVE_INTERVAL = 10.0  # seconds between live-table refreshes


@dataclass
class Task:
    """A fully-resolved ``compute`` request handed to :func:`run`.

    The dataset(s) are described by picklable ``open_dataset`` specs (positional
    args + kwargs) so that worker processes can re-open them.
    """

    open_args: list[Any]
    open_kwargs: dict[str, Any]
    label: str
    do_statistics: bool = False
    tendency: str | None = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    allow_nans: bool = False
    has_residual: bool = False
    residual_open_args: list[Any] = field(default_factory=list)
    residual_open_kwargs: dict[str, Any] = field(default_factory=dict)
    residual_label: str = ""
    parallel: int = 0
    checkpoint_path: str | None = None
    resume: bool = False
    args_sha: str = ""
    sample_dates: float | None = None
    live: bool = False
    # Optional interpolation: both datasets are brought onto this grid before
    # they are read, differenced and accumulated.
    grid: str | None = None
    grid_method: str = DEFAULT_GRID_METHOD
    # Optional dataset output: the values read by the loop (the dataset as
    # opened, or the residual) are also written to this zarr store.
    output_dataset: str | None = None
    output_overwrite: bool = False


class Collectors:
    """Bundle of the statistics + (optional) tendency accumulator for one computation.

    Parameters
    ----------
    variables : list of str
        Variable names.
    do_statistics : bool
        Whether to collect plain statistics.
    tendency_steps : int or None
        The tendency delta in time steps, or ``None`` for no tendency.
    allow_nans : bool
        NaN policy passed to the accumulators.
    track_constants : bool, optional
        Whether to also track which variables are constant in time. Only needed
        when a dataset is generated, whose metadata records them.
    """

    def __init__(
        self,
        variables: list[str],
        do_statistics: bool,
        tendency_steps: int | None,
        allow_nans: bool,
        track_constants: bool = False,
    ) -> None:
        self.variables = list(variables)
        self.stats = Accumulator(variables, allow_nans=allow_nans) if do_statistics else None
        self.tend = (
            None if tendency_steps is None else TendencyAccumulator(variables, tendency_steps, allow_nans=allow_nans)
        )
        self.consts = ConstantsTracker(variables) if track_constants else None

    def seed(self, seed_data: NDArray[Any]) -> None:
        """Seed the tendency accumulator's window (parallel boundary handling)."""
        if self.tend is not None:
            self.tend.seed_window(seed_data)

    def new_run(self) -> None:
        """Start a new run of consecutive dates: forget the tendency window.

        Called when the next block does not immediately follow the previous one,
        i.e. across a missing date. Tendencies straddling the gap would otherwise
        be computed between dates that are not ``delta`` apart.
        """
        if self.tend is not None:
            self.tend.reset_window()

    def update(self, data: NDArray[Any]) -> None:
        """Feed a chunk of data to the accumulators."""
        if self.stats is not None:
            self.stats.update(data)
        if self.tend is not None:
            self.tend.update(data)
        if self.consts is not None:
            self.consts.update(data)

    def merge(self, other: "Collectors") -> "Collectors":
        """Merge another bundle into a new one (used by the parallel path)."""
        result = Collectors.__new__(Collectors)
        result.variables = self.variables
        result.stats = None if self.stats is None else self.stats.merge(other.stats)
        result.tend = None if self.tend is None else self.tend.merge(other.tend)
        result.consts = None if self.consts is None else self.consts.merge(other.consts)
        return result

    def results(self) -> dict[str, Any]:
        """Return ``{"statistics": ..., "tendency": ..., "constants": ...}`` (each may be ``None``)."""
        return {
            "statistics": None if self.stats is None else self.stats.statistics(),
            "tendency": None if self.tend is None else self.tend.statistics(),
            "constants": self.consts,
        }


def _read_block(
    ds_a: Any,
    ds_b: Any,
    block: Any,
    interpolators: tuple[Interpolator | None, Interpolator | None] = (None, None),
) -> NDArray[np.float64]:
    """Read a block, interpolating and subtracting as requested.

    This is the single place where the values fed to everything else are formed:
    each dataset is read, interpolated onto the target grid when ``--grid`` is
    given, and B is subtracted from A when ``--minus`` is given. Statistics,
    tendencies and the generated dataset are then blind to what happened here.

    Parameters
    ----------
    ds_a : Dataset
        The dataset to read.
    ds_b : Dataset or None
        The dataset to subtract, or ``None``.
    block : slice or list of int
        The time range (or the list of time indices) to read.
    interpolators : tuple
        The interpolators of A and B; each may be ``None``.

    Returns
    -------
    ndarray
        The values, in ``float64``.
    """
    interpolator_a, interpolator_b = interpolators

    a = np.asarray(ds_a[block], dtype=np.float64)
    if interpolator_a is not None:
        a = interpolator_a(a)

    if ds_b is None:
        return a

    b = np.asarray(ds_b[block], dtype=np.float64)
    if interpolator_b is not None:
        b = interpolator_b(b)

    return a - b


def _read(
    ds_a: Any,
    ds_b: Any,
    lo: int,
    hi: int,
    interpolators: tuple[Interpolator | None, Interpolator | None] = (None, None),
) -> NDArray[np.float64]:
    """Read ``[lo:hi]``, interpolating and subtracting as requested."""
    return _read_block(ds_a, ds_b, slice(lo, hi), interpolators)


def _seed_from_sha(args_sha: str) -> int:
    """Derive a deterministic integer seed from the arguments hash (or any string)."""
    import zlib

    return zlib.crc32((args_sha or "").encode())


def _sample_indices(n: int, fraction: float, seed: int, missing: frozenset[int] = frozenset()) -> NDArray[np.int64]:
    """Return a sorted random sample of a fraction of the readable time indices.

    Missing dates are excluded from the pool the sample is drawn from, so the
    fraction is of the dates that can actually be read.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"--sample-dates fraction must be in (0, 1], got {fraction}")
    candidates = np.arange(n) if not missing else np.array(sorted(set(range(n)) - missing), dtype=np.int64)
    if len(candidates) == 0:
        raise ValueError("No date can be read: they are all missing")
    k = max(1, round(len(candidates) * fraction))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=k, replace=False))


def _missing(ds_a: Any, ds_b: Any) -> frozenset[int]:
    """Return the time indices that cannot be read from either dataset.

    Missing dates are recorded in the store and raise ``MissingDateError`` when
    read, so they are skipped by the computation. Both datasets are indexed by
    position by ``--minus``, so a date missing from either one is missing from
    their difference.

    Parameters
    ----------
    ds_a : Dataset
        The opened dataset.
    ds_b : Dataset or None
        The dataset to subtract, or ``None``.

    Returns
    -------
    frozenset of int
        The indices to skip.
    """
    missing = set(ds_a.missing)
    if ds_b is not None:
        missing |= set(ds_b.missing)
    return frozenset(missing)


def _runs(start: int, end: int, missing: frozenset[int]) -> list[tuple[int, int]]:
    """Split ``[start, end)`` into the maximal runs of consecutive readable dates.

    Parameters
    ----------
    start, end : int
        The range to split.
    missing : frozenset of int
        The indices to leave out.

    Returns
    -------
    list of tuple of (int, int)
        The ``[lo, hi)`` runs, in order; empty runs are not returned.
    """
    if not missing:
        return [(start, end)] if end > start else []

    runs = []
    lo = start
    for i in sorted(m for m in missing if start <= m < end):
        if i > lo:
            runs.append((lo, i))
        lo = i + 1
    if end > lo:
        runs.append((lo, end))
    return runs


def _blocks(
    n: int,
    chunk_size: int,
    indices: NDArray[np.int64] | None,
    missing: frozenset[int] = frozenset(),
) -> list[tuple[Any, bool]]:
    """Build the deterministic list of read blocks.

    Each block is a ``slice`` (contiguous, full mode) or a list of integer indices
    (subsampled mode). The list is deterministic so a resumed run reproduces it.
    Missing dates are left out, which splits the contiguous range into runs; the
    flag returned with each block says whether it starts a new one, so that the
    caller can drop the tendency window rather than compute a tendency across the
    gap.

    Parameters
    ----------
    n : int
        Dataset length.
    chunk_size : int
        Number of time steps per block.
    indices : ndarray or None
        Sampled indices, or ``None`` for the full contiguous range.
    missing : frozenset of int, optional
        Time indices that cannot be read.

    Returns
    -------
    list of tuple
        ``(block, starts_run)`` pairs, where ``block`` is a ``slice`` or a list of
        integer indices.
    """
    chunk_size = max(1, int(chunk_size))

    if indices is None:
        blocks: list[tuple[Any, bool]] = []
        for run_lo, run_hi in _runs(0, n, missing):
            for k, lo in enumerate(range(run_lo, run_hi, chunk_size)):
                blocks.append((slice(lo, min(lo + chunk_size, run_hi)), k == 0))
        return blocks

    # Subsampled: the dates are not adjacent anyway, so tendencies are refused
    # upstream and every block is flagged as starting a run.
    kept = [i for i in indices if int(i) not in missing]
    return [(list(kept[i : i + chunk_size]), True) for i in range(0, len(kept), chunk_size)]


def _render_live(
    collectors: "Collectors", variables: list[str], idxs: list[int], prev: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Print a snapshot of the current statistics for the given variables.

    Mirrors the ``inspect`` command's statistics table (Index/Variable/Min/Max/
    Mean/Stdev). Each numeric cell also shows the signed change since the previous
    refresh, e.g. ``0.707 (+0.001)``. Written via :func:`tqdm.write` so the
    progress bar is preserved.

    Parameters
    ----------
    collectors : Collectors
        The running accumulators.
    variables : list of str
        All variable names.
    idxs : list of int
        Indices of the variables to display.
    prev : dict or None
        The statistics returned by the previous call, used to compute the deltas
        (``None`` on the first refresh).

    Returns
    -------
    dict or None
        The current statistics, to be passed back as ``prev`` next time.
    """
    acc = collectors.stats if collectors.stats is not None else collectors.tend
    if acc is None:
        return prev
    stats = acc.statistics()

    import io
    import math

    import tqdm
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Statistics (live)")
    for col in ("Index", "Variable", "Min", "Max", "Mean", "Stdev"):
        table.add_column(col, justify="left" if col == "Variable" else "right")

    def _f(x: Any) -> str:
        return f"{float(x):.3g}"

    def _cell(key: str, i: int) -> str:
        v = float(stats[key][i])
        if prev is None:
            return _f(v)
        p = float(prev[key][i])
        d = v - p
        # Prefer a relative change, but fall back to the absolute delta when the
        # baseline is zero/non-finite (avoids division by zero) or when the change
        # exceeds +/-100% (percentages become meaningless / hard to read).
        pct = d / p * 100.0 if p != 0 else float("nan")
        if not math.isfinite(pct) or abs(pct) > 100.0:
            return f"{_f(v)} ({d:+.3g})"
        return f"{_f(v)} ({pct:+.3g}%)"

    for i in idxs:
        table.add_row(
            str(i),
            variables[i],
            _cell("minimum", i),
            _cell("maximum", i),
            _cell("mean", i),
            _cell("stdev", i),
        )

    buffer = io.StringIO()
    Console(file=buffer, width=120).print(table)
    tqdm.tqdm.write(buffer.getvalue())
    return stats


def _open(open_args: list[Any], open_kwargs: dict[str, Any]) -> Any:
    """Open a dataset from a picklable spec."""
    from anemoi.datasets import open_dataset

    return open_dataset(*open_args, **open_kwargs)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def _save_checkpoint(path: str, payload: dict[str, Any]) -> None:
    """Atomically write a checkpoint to ``path``."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)
    LOG.info("Checkpoint written to %s (%s)", path, payload.get("progress", ""))


def _load_checkpoint(path: str, args_sha: str) -> dict[str, Any] | None:
    """Load and validate a checkpoint, or return ``None`` if unusable."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("version") != CHECKPOINT_VERSION:
        LOG.warning("Ignoring checkpoint %s: version mismatch", path)
        return None
    if payload.get("args_sha") != args_sha:
        raise ValueError(
            f"Checkpoint {path} was produced with different arguments "
            f"(sha {payload.get('args_sha')} != {args_sha}). Delete it or use a fresh --checkpoint."
        )
    return payload


# --------------------------------------------------------------------------- #
# Sequential
# --------------------------------------------------------------------------- #


def _run_sequential(
    task: Task,
    ds_a: Any,
    ds_b: Any,
    variables: list[str],
    tendency_steps: int | None,
    indices: NDArray[np.int64] | None,
    output: OutputDataset | None = None,
    interpolators: tuple[Interpolator | None, Interpolator | None] = (None, None),
    missing: frozenset[int] = frozenset(),
) -> Collectors:
    """Run the computation in-process with a time-based checkpoint and live table.

    Iterates over deterministic blocks (contiguous slices, or lists of sampled
    indices), leaving out the missing dates. When ``task.live`` is set, a
    statistics table for all variables is refreshed every :data:`LIVE_INTERVAL`
    seconds. When ``output`` is given, each block is also written to the generated
    dataset.
    """
    import tqdm

    n = len(ds_a)
    blocks = _blocks(n, task.chunk_size, indices, missing)
    total = len(blocks)

    collectors: Collectors | None = None
    next_block = 0

    payload = _load_checkpoint(task.checkpoint_path, task.args_sha) if task.resume else None
    if payload is not None and payload.get("mode") == "sequential":
        collectors = payload["collectors"]
        next_block = payload["next_block"]
        LOG.info("Resuming sequential computation from block %d/%d", next_block, total)

    if collectors is None:
        collectors = Collectors(
            variables,
            task.do_statistics,
            tendency_steps,
            task.allow_nans,
            track_constants=output is not None,
        )

    if output is not None:
        output.open_for_write()

    # The live table shows every variable.
    live_idxs: list[int] = list(range(len(variables))) if task.live else []

    last_ckpt = time.time()
    last_live = time.time()
    prev_live: dict[str, Any] | None = None
    bar = tqdm.tqdm(range(next_block, total), desc="compute", initial=next_block, total=total)
    for b in bar:
        block, starts_run = blocks[b]
        if starts_run:
            collectors.new_run()
        data = _read_block(ds_a, ds_b, block, interpolators)
        if output is not None:
            output.write(block, data)
        collectors.update(data)

        now = time.time()
        if task.live and live_idxs and now - last_live > LIVE_INTERVAL:
            prev_live = _render_live(collectors, variables, live_idxs, prev_live)
            last_live = now

        if task.checkpoint_path and now - last_ckpt > CHECKPOINT_INTERVAL:
            _save_checkpoint(
                task.checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "args_sha": task.args_sha,
                    "mode": "sequential",
                    "collectors": collectors,
                    "next_block": b + 1,
                    "progress": f"{b + 1}/{total} blocks",
                },
            )
            last_ckpt = now

    return collectors


# --------------------------------------------------------------------------- #
# Parallel
# --------------------------------------------------------------------------- #


def _segments(n: int, chunk_size: int, workers: int) -> list[tuple[int, int]]:
    """Split ``[0, n)`` into roughly ``workers * 4`` chunk-aligned segments."""
    n_segments = max(workers * 4, workers)
    seg_len = max(chunk_size, -(-n // n_segments))  # ceil division
    return [(s, min(s + seg_len, n)) for s in range(0, n, seg_len)]


def _run_segment(
    task: Task,
    variables: list[str],
    tendency_steps: int | None,
    seg_start: int,
    seg_end: int,
) -> Collectors:
    """Worker entry point: compute one segment and return its accumulators.

    The whole :class:`Task` is passed because it is picklable and holds everything
    the worker needs to rebuild its own state: the dataset(s) are re-opened from
    their ``open_dataset`` specs (datasets themselves are not picklable), and so
    are the interpolators and the writer of the generated dataset.

    When a tendency is requested and the segment does not start at 0, the
    ``tendency_steps`` rows before ``seg_start`` are read to seed the sliding
    window so that boundary tendencies are computed correctly (those rows belong
    to the previous segment and are not written to a generated dataset). The
    seeding stops at the last missing date before the segment, so that the window
    holds exactly the rows the sequential loop would have kept and no tendency is
    computed across a gap.

    When a dataset is generated, the worker writes its own segment. Segments are
    chunk-aligned and the output has one date per chunk, so no two workers ever
    write to the same chunk.

    Parameters
    ----------
    task : Task
        The fully-resolved request.
    variables : list of str
        Variable names.
    tendency_steps : int or None
        The tendency delta in time steps.
    seg_start, seg_end : int
        The time range of the segment.

    Returns
    -------
    Collectors
        The accumulators of this segment.
    """
    ds_a = _open(task.open_args, task.open_kwargs)
    ds_b = _open(task.residual_open_args, task.residual_open_kwargs) if task.has_residual else None

    interpolators = _interpolators(task, ds_a, ds_b)
    missing = _missing(ds_a, ds_b)

    output = None
    if task.output_dataset:
        output = OutputDataset(task.output_dataset).open_for_write()

    collectors = Collectors(
        variables,
        task.do_statistics,
        tendency_steps,
        task.allow_nans,
        track_constants=output is not None,
    )

    if tendency_steps and seg_start > 0:
        # The rows of the previous segment the sliding window would hold here: the
        # last `tendency_steps` ones, but never reaching back across a missing date.
        gap = max((m for m in missing if m < seg_start), default=-1)
        s0 = max(seg_start - tendency_steps, gap + 1, 0)
        if s0 < seg_start:
            collectors.seed(_read(ds_a, ds_b, s0, seg_start, interpolators))

    for run_lo, run_hi in _runs(seg_start, seg_end, missing):
        first = True
        for lo, hi in iter_chunks(run_hi, run_lo, run_hi, task.chunk_size):
            if first:
                # Only after a gap: at the segment start the window was just seeded.
                if run_lo != seg_start:
                    collectors.new_run()
                first = False
            data = _read(ds_a, ds_b, lo, hi, interpolators)
            if output is not None:
                output.write(slice(lo, hi), data)
            collectors.update(data)

    return collectors


def _run_parallel(
    task: Task,
    ds_a: Any,
    variables: list[str],
    tendency_steps: int | None,
    output: OutputDataset | None = None,
) -> Collectors:
    """Run the computation across ``task.parallel`` worker processes."""
    n = len(ds_a)
    segments = _segments(n, task.chunk_size, task.parallel)

    merged: Collectors | None = None
    completed: set[int] = set()

    payload = _load_checkpoint(task.checkpoint_path, task.args_sha) if task.resume else None
    if payload is not None and payload.get("mode") == "parallel":
        merged = payload["collectors"]
        completed = payload["completed"]
        LOG.info("Resuming parallel computation: %d/%d segments already done", len(completed), len(segments))

    todo = [(i, s, e) for i, (s, e) in enumerate(segments) if i not in completed]
    LOG.info("Computing %d segments across %d workers", len(todo), task.parallel)

    import tqdm

    with ProcessPoolExecutor(max_workers=task.parallel) as pool:
        futures = {pool.submit(_run_segment, task, variables, tendency_steps, s, e): i for (i, s, e) in todo}
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="segments"):
            seg_id = futures[fut]
            collectors = fut.result()
            merged = collectors if merged is None else merged.merge(collectors)
            completed.add(seg_id)
            if task.checkpoint_path:
                _save_checkpoint(
                    task.checkpoint_path,
                    {
                        "version": CHECKPOINT_VERSION,
                        "args_sha": task.args_sha,
                        "mode": "parallel",
                        "collectors": merged,
                        "completed": completed,
                        "progress": f"{len(completed)}/{len(segments)} segments",
                    },
                )

    if merged is None:  # everything was already completed on resume
        merged = Collectors(
            variables,
            task.do_statistics,
            tendency_steps,
            task.allow_nans,
            track_constants=output is not None,
        )
    return merged


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(task: Task) -> tuple[list[str], dict[str, Any]]:
    """Execute a :class:`Task` and return ``(variables, results)``.

    When ``task.output_dataset`` is set, the values read by the loop -- the dataset
    as opened, or the residual -- are also written to that zarr store, which is
    finalised with the computed statistics as its own.

    Parameters
    ----------
    task : Task
        The fully-resolved request.

    Returns
    -------
    tuple
        ``(variables, results)`` where ``results`` is
        ``{"statistics": {...}|None, "tendency": {...}|None, "constants": ...,
        "derived_from": {...}}``.
    """
    LOG.info("Opening dataset %s", task.label)
    ds_a = _open(task.open_args, task.open_kwargs)
    variables = list(ds_a.variables)

    tendency_steps = None
    if task.tendency is not None:
        if ds_a.frequency is None:
            # Trajectory datasets have two frequencies (base date and forecast
            # step), so 'the previous date' is ambiguous.
            raise ValueError(
                f"--statistics-tendencies is not supported for '{task.label}': it has two time axes "
                "(base dates and forecast steps) and therefore no single frequency."
            )
        tendency_steps = delta_to_steps(task.tendency, ds_a.frequency)

    ds_b = None
    if task.has_residual:
        LOG.info("Opening the dataset to subtract: %s", task.residual_label)
        ds_b = _open(task.residual_open_args, task.residual_open_kwargs)
        # With --grid both datasets end up on the same grid, so their grids are
        # allowed to differ here.
        _check_compatible(ds_a, ds_b, ignore_grid=task.grid is not None)

    interpolators = _interpolators(task, ds_a, ds_b)

    # Missing dates cannot be read at all: they are left out of the computation
    # (of both datasets, since --minus subtracts by position).
    missing = _missing(ds_a, ds_b)
    if missing:
        LOG.warning(
            "Skipping %d missing date(s) out of %d. Open the dataset with 'fill_missing_dates=...' to include them.",
            len(missing),
            len(ds_a),
        )
        if len(missing) == len(ds_a):
            raise ValueError(f"All {len(ds_a)} dates of {task.label} are missing; there is nothing to compute.")

    # Date subsampling: only valid for plain/residual statistics in sequential mode.
    indices = None
    if task.sample_dates is not None:
        if tendency_steps is not None:
            raise ValueError(
                "--sample-dates cannot be combined with --statistics-tendencies (tendencies need adjacent dates)"
            )
        if task.parallel and task.parallel > 1:
            raise ValueError("--sample-dates is not supported with --parallel; run sequentially")
        indices = _sample_indices(len(ds_a), task.sample_dates, _seed_from_sha(task.args_sha), missing)
        LOG.info("Subsampling %d/%d dates (%.1f%%)", len(indices), len(ds_a) - len(missing), 100 * task.sample_dates)

    # How the values were obtained: the open_dataset call(s), the arithmetic and
    # the versions. It goes both into the statistics document and, when one is
    # generated, into the dataset's metadata, so the two cannot disagree.
    provenance = derived_from(
        task.label,
        task.open_args,
        task.open_kwargs,
        ds_a,
        residual_label=task.residual_label if ds_b is not None else None,
        residual_open_args=task.residual_open_args if ds_b is not None else None,
        residual_open_kwargs=task.residual_open_kwargs if ds_b is not None else None,
        residual_dataset=ds_b,
        tendency=task.tendency,
        chunk_size=task.chunk_size,
        allow_nans=task.allow_nans,
        grid=task.grid,
        grid_method=task.grid_method if task.grid else None,
    )

    output = _prepare_output(task, ds_a, ds_b, interpolators[0], missing, provenance)

    if task.parallel and task.parallel > 1:
        collectors = _run_parallel(task, ds_a, variables, tendency_steps, output)
    else:
        collectors = _run_sequential(
            task, ds_a, ds_b, variables, tendency_steps, indices, output, interpolators, missing
        )

    results = collectors.results()
    results["derived_from"] = provenance

    # Computation finished cleanly: drop the checkpoint so a later run starts fresh.
    if task.checkpoint_path and os.path.exists(task.checkpoint_path):
        try:
            os.remove(task.checkpoint_path)
            LOG.info("Removed checkpoint %s (computation complete)", task.checkpoint_path)
        except OSError:
            pass

    if output is not None:
        output.finalise(variables, results, tendency=task.tendency)

    return variables, results


def _interpolators(task: Task, ds_a: Any, ds_b: Any) -> tuple[Interpolator | None, Interpolator | None]:
    """Build the interpolators of both datasets for a ``--grid`` request.

    Parameters
    ----------
    task : Task
        The fully-resolved request.
    ds_a : Dataset
        The opened dataset.
    ds_b : Dataset or None
        The dataset to subtract, if any.

    Returns
    -------
    tuple
        The interpolators of A and B; each is ``None`` when there is nothing to
        do (no ``--grid``, or the dataset is already on the target grid).
    """
    if not task.grid:
        return (None, None)

    target = TargetGrid(task.grid)
    return (
        Interpolator.build(ds_a, target, task.grid_method),
        None if ds_b is None else Interpolator.build(ds_b, target, task.grid_method),
    )


def _prepare_output(
    task: Task,
    ds_a: Any,
    ds_b: Any,
    interpolator: Interpolator | None = None,
    missing: frozenset[int] = frozenset(),
    provenance: dict[str, Any] | None = None,
) -> OutputDataset | None:
    """Create the dataset to generate, if one was requested.

    Parameters
    ----------
    task : Task
        The fully-resolved request.
    ds_a : Dataset
        The opened dataset.
    ds_b : Dataset or None
        The subtracted dataset, for a residual.
    interpolator : Interpolator or None
        The interpolator of ``ds_a``, when ``--grid`` was given. The generated
        dataset is then on the target grid rather than on the dataset's own.
    missing : frozenset of int, optional
        The time indices that cannot be read; the generated dataset records them
        as its own missing dates.
    provenance : dict, optional
        The ``derived_from`` entry describing where the values come from, as
        built by :func:`derived_from`.

    Returns
    -------
    OutputDataset or None
        The writer, or ``None`` when no dataset is generated.
    """
    if not task.output_dataset:
        return None

    if task.sample_dates is not None:
        raise ValueError(
            "A generated dataset cannot be combined with --sample-dates: every date must be read to be written"
        )

    if not task.do_statistics:
        raise ValueError("A generated dataset needs its statistics; --statistics cannot be turned off")

    output = OutputDataset(task.output_dataset)

    OutputDataset.check_dataset(ds_a, task.label)
    if ds_b is not None:
        OutputDataset.check_dataset(ds_b, task.residual_label)

    if task.resume and os.path.exists(task.output_dataset):
        # Resuming: the blocks recorded in the checkpoint have already been
        # written, so keep the store (and its metadata) as it is.
        LOG.info("Resuming: writing into the existing dataset %s", task.output_dataset)
        return output

    output.create(
        ds_a,
        derived_from=provenance,
        residual=ds_b is not None,
        allow_nans=task.allow_nans,
        overwrite=task.output_overwrite,
        grid=None if interpolator is None else interpolator.target,
        missing=missing,
    )
    return output
