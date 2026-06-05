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

from .statistics import DEFAULT_CHUNK_SIZE
from .statistics import Accumulator
from .statistics import iter_chunks
from .statistics_residuals import _check_compatible
from .statistics_tendencies import TendencyAccumulator
from .statistics_tendencies import delta_to_steps

LOG = logging.getLogger(__name__)

CHECKPOINT_VERSION = 2
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
    """

    def __init__(self, variables: list[str], do_statistics: bool, tendency_steps: int | None, allow_nans: bool) -> None:
        self.variables = list(variables)
        self.stats = Accumulator(variables, allow_nans=allow_nans) if do_statistics else None
        self.tend = (
            None if tendency_steps is None else TendencyAccumulator(variables, tendency_steps, allow_nans=allow_nans)
        )

    def seed(self, seed_data: NDArray[Any]) -> None:
        """Seed the tendency accumulator's window (parallel boundary handling)."""
        if self.tend is not None:
            self.tend.seed_window(seed_data)

    def update(self, data: NDArray[Any]) -> None:
        """Feed a chunk of data to the accumulators."""
        if self.stats is not None:
            self.stats.update(data)
        if self.tend is not None:
            self.tend.update(data)

    def merge(self, other: "Collectors") -> "Collectors":
        """Merge another bundle into a new one (used by the parallel path)."""
        result = Collectors.__new__(Collectors)
        result.variables = self.variables
        result.stats = None if self.stats is None else self.stats.merge(other.stats)
        result.tend = None if self.tend is None else self.tend.merge(other.tend)
        return result

    def results(self) -> dict[str, Any]:
        """Return ``{"statistics": {...}|None, "tendency": {...}|None}``."""
        return {
            "statistics": None if self.stats is None else self.stats.statistics(),
            "tendency": None if self.tend is None else self.tend.statistics(),
        }


def _read(ds_a: Any, ds_b: Any, lo: int, hi: int) -> NDArray[np.float64]:
    """Read ``[lo:hi]`` from dataset A, subtracting B when computing a residual."""
    a = np.asarray(ds_a[lo:hi], dtype=np.float64)
    if ds_b is None:
        return a
    return a - np.asarray(ds_b[lo:hi], dtype=np.float64)


def _read_block(ds_a: Any, ds_b: Any, block: Any) -> NDArray[np.float64]:
    """Read a block (a ``slice`` or a list of indices), subtracting B for residuals."""
    a = np.asarray(ds_a[block], dtype=np.float64)
    if ds_b is None:
        return a
    return a - np.asarray(ds_b[block], dtype=np.float64)


def _seed_from_sha(args_sha: str) -> int:
    """Derive a deterministic integer seed from the arguments hash (or any string)."""
    import zlib

    return zlib.crc32((args_sha or "").encode())


def _sample_indices(n: int, fraction: float, seed: int) -> NDArray[np.int64]:
    """Return a sorted random sample of ``round(n * fraction)`` time indices."""
    if not 0 < fraction <= 1:
        raise ValueError(f"--sample-dates fraction must be in (0, 1], got {fraction}")
    k = max(1, round(n * fraction))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


def _blocks(n: int, chunk_size: int, indices: NDArray[np.int64] | None) -> list[Any]:
    """Build the deterministic list of read blocks.

    Each block is a ``slice`` (contiguous, full mode) or a list of integer indices
    (subsampled mode). The list is deterministic so a resumed run reproduces it.

    Parameters
    ----------
    n : int
        Dataset length.
    chunk_size : int
        Number of time steps per block.
    indices : ndarray or None
        Sampled indices, or ``None`` for the full contiguous range.
    """
    chunk_size = max(1, int(chunk_size))
    if indices is None:
        return [slice(lo, min(lo + chunk_size, n)) for lo in range(0, n, chunk_size)]
    return [list(indices[i : i + chunk_size]) for i in range(0, len(indices), chunk_size)]


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
) -> Collectors:
    """Run the computation in-process with a time-based checkpoint and live table.

    Iterates over deterministic blocks (contiguous slices, or lists of sampled
    indices). When ``task.live`` is set, a statistics table for all variables is
    refreshed every :data:`LIVE_INTERVAL` seconds.
    """
    import tqdm

    n = len(ds_a)
    blocks = _blocks(n, task.chunk_size, indices)
    total = len(blocks)

    collectors: Collectors | None = None
    next_block = 0

    payload = _load_checkpoint(task.checkpoint_path, task.args_sha) if task.resume else None
    if payload is not None and payload.get("mode") == "sequential":
        collectors = payload["collectors"]
        next_block = payload["next_block"]
        LOG.info("Resuming sequential computation from block %d/%d", next_block, total)

    if collectors is None:
        collectors = Collectors(variables, task.do_statistics, tendency_steps, task.allow_nans)

    # The live table shows every variable.
    live_idxs: list[int] = list(range(len(variables))) if task.live else []

    last_ckpt = time.time()
    last_live = time.time()
    prev_live: dict[str, Any] | None = None
    bar = tqdm.tqdm(range(next_block, total), desc="compute", initial=next_block, total=total)
    for b in bar:
        collectors.update(_read_block(ds_a, ds_b, blocks[b]))

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
    spec_a: tuple[list[Any], dict[str, Any]],
    spec_b: tuple[list[Any], dict[str, Any]] | None,
    variables: list[str],
    do_statistics: bool,
    tendency_steps: int | None,
    allow_nans: bool,
    chunk_size: int,
    seg_start: int,
    seg_end: int,
) -> Collectors:
    """Worker entry point: compute one segment and return its accumulators.

    Re-opens the dataset(s) from their specs (datasets are not picklable). When a
    tendency is requested and the segment does not start at 0, the ``tendency_steps``
    rows before ``seg_start`` are read to seed the sliding window so that boundary
    tendencies are computed correctly.
    """
    ds_a = _open(*spec_a)
    ds_b = _open(*spec_b) if spec_b is not None else None

    collectors = Collectors(variables, do_statistics, tendency_steps, allow_nans)

    if tendency_steps and seg_start > 0:
        s0 = max(0, seg_start - tendency_steps)
        collectors.seed(_read(ds_a, ds_b, s0, seg_start))

    for lo, hi in iter_chunks(seg_end, seg_start, seg_end, chunk_size):
        collectors.update(_read(ds_a, ds_b, lo, hi))

    return collectors


def _run_parallel(task: Task, ds_a: Any, variables: list[str], tendency_steps: int | None) -> Collectors:
    """Run the computation across ``task.parallel`` worker processes."""
    n = len(ds_a)
    segments = _segments(n, task.chunk_size, task.parallel)

    spec_a = (task.open_args, task.open_kwargs)
    spec_b = (task.residual_open_args, task.residual_open_kwargs) if task.has_residual else None

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
        futures = {
            pool.submit(
                _run_segment,
                spec_a,
                spec_b,
                variables,
                task.do_statistics,
                tendency_steps,
                task.allow_nans,
                task.chunk_size,
                s,
                e,
            ): i
            for (i, s, e) in todo
        }
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
        merged = Collectors(variables, task.do_statistics, tendency_steps, task.allow_nans)
    return merged


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run(task: Task) -> tuple[list[str], dict[str, Any]]:
    """Execute a :class:`Task` and return ``(variables, results)``.

    Parameters
    ----------
    task : Task
        The fully-resolved request.

    Returns
    -------
    tuple
        ``(variables, results)`` where ``results`` is
        ``{"statistics": {...}|None, "tendency": {...}|None}``.
    """
    LOG.info("Opening dataset %s", task.label)
    ds_a = _open(task.open_args, task.open_kwargs)
    variables = list(ds_a.variables)

    tendency_steps = None if task.tendency is None else delta_to_steps(task.tendency, ds_a.frequency)

    ds_b = None
    if task.has_residual:
        LOG.info("Opening residual dataset %s", task.residual_label)
        ds_b = _open(task.residual_open_args, task.residual_open_kwargs)
        _check_compatible(ds_a, ds_b)

    # Date subsampling: only valid for plain/residual statistics in sequential mode.
    indices = None
    if task.sample_dates is not None:
        if tendency_steps is not None:
            raise ValueError("--sample-dates cannot be combined with --statistics-tendencies (tendencies need adjacent dates)")
        if task.parallel and task.parallel > 1:
            raise ValueError("--sample-dates is not supported with --parallel; run sequentially")
        indices = _sample_indices(len(ds_a), task.sample_dates, _seed_from_sha(task.args_sha))
        LOG.info("Subsampling %d/%d dates (%.1f%%)", len(indices), len(ds_a), 100 * task.sample_dates)

    if task.parallel and task.parallel > 1:
        collectors = _run_parallel(task, ds_a, variables, tendency_steps)
    else:
        collectors = _run_sequential(task, ds_a, ds_b, variables, tendency_steps, indices)

    results = collectors.results()

    # Computation finished cleanly: drop the checkpoint so a later run starts fresh.
    if task.checkpoint_path and os.path.exists(task.checkpoint_path):
        try:
            os.remove(task.checkpoint_path)
            LOG.info("Removed checkpoint %s (computation complete)", task.checkpoint_path)
        except OSError:
            pass

    return variables, results
