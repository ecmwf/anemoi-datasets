# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Two-step read: collect → factorize → execute.

See docs/adr/adr-3-two-step-read.md for the full design.

Usage
-----
Instead of ``ds[n]`` calling zarr immediately through the wrapper chain,
``two_step_read(ds, n)`` does:

  1. ``ds.collect_read_parts(n)`` — walks the wrapper tree, returns all
     :class:`ReadPart` objects (no I/O).
  2. :func:`factorize` — merges parts that share the same zarr array and
     non-date dimensions into a single bounding-box read.
  3. :func:`execute_parts` — sequential zarr reads (cross-part threading is
     GIL-bound on decompress; within-array S3 fetch is already parallel).
  4. ``ds.read_from_buffer(n, buffer)`` — reassembles the result from cached
     arrays, applying all wrapper transformations.

Enable debug logging with the environment variable::

    ANEMOI_DATASETS_READ_PARTS_DEBUG=1

or by setting ``logging.DEBUG`` on the ``anemoi.datasets.read_parts`` logger.
"""

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from anemoi.datasets.usage.dataset import Dataset
    from anemoi.datasets.usage.dataset import FullIndex

LOG = logging.getLogger("anemoi.datasets.read_parts")

READ_PARTS_DEBUG = os.environ.get("ANEMOI_DATASETS_READ_PARTS_DEBUG", "").lower() in ("1", "true", "yes")
# Two-step read is on by default; set ANEMOI_DATASETS_READ_PARTS=0 (or false/no)
# to force the eager recursive __getitem__ path (debugging / kill-switch).
READ_PARTS_ENABLED = os.environ.get("ANEMOI_DATASETS_READ_PARTS", "1").lower() not in ("0", "false", "no")

# Normalised slice stored as (start, stop, step) — all concrete ints, no None.
NormSlice = tuple[int, int, int]


class ReadPart:
    """One rectangular read from a zarr data array.

    Parameters
    ----------
    path : str
        Human-readable path label (for logging/repr only).
    data : zarr.Array
        The zarr array to read from.  Kept as a reference so ``execute``
        works for both on-disk and in-memory (test) stores without needing
        to re-open by path.
    slices : tuple of NormSlice
        One ``(start, stop, step)`` per zarr dimension, all concrete.
    squeeze : tuple of int
        Axes that were originally integer indices; squeezed after the read.
    grid_index : tuple of int, optional
        If set, the **last** axis is selected by this index array (zarr
        orthogonal indexing) instead of by ``slices[-1]``, which becomes a
        placeholder.  Used for grid-subset pushdown (e.g. ``Cutout`` reading
        only the masked/sharded grid points it needs).  ``None`` means a plain
        rectangular read.
    """

    __slots__ = ("path", "data", "slices", "squeeze", "grid_index")

    def __init__(
        self,
        path: str,
        data: Any,
        slices: tuple[NormSlice, ...],
        squeeze: tuple[int, ...],
        grid_index: tuple[int, ...] | None = None,
    ) -> None:
        self.path = path
        self.data = data
        self.slices = slices
        self.squeeze = squeeze
        self.grid_index = grid_index

    @classmethod
    def from_raw_slices(
        cls,
        path: str,
        data: Any,
        slices: tuple[slice, ...],
        squeeze: tuple[int, ...],
        grid_index: tuple[int, ...] | None = None,
    ) -> "ReadPart":
        """Build from a tuple of concrete ``slice`` objects."""
        norm: tuple[NormSlice, ...] = tuple(
            (s.start, s.stop, s.step) for s in slices
        )
        return cls(path=path, data=data, slices=norm, squeeze=squeeze, grid_index=grid_index)

    def to_zarr_index(self) -> tuple[slice, ...]:
        """Convert stored slices back to zarr-compatible ``slice`` objects."""
        return tuple(slice(s, e, t) for s, e, t in self.slices)

    def execute(self) -> NDArray:
        """Execute the zarr read and return the array."""
        if self.grid_index is None:
            return self.data[self.to_zarr_index()]
        # Orthogonal selection: slices on every axis but the last, an index
        # array on the last (grid) axis.  Only the needed grid points are read.
        selection: tuple = self.to_zarr_index()[:-1] + (np.asarray(self.grid_index),)
        return self.data.oindex[selection]

    def _identity(self) -> tuple:
        return (id(self.data), self.slices, self.grid_index)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReadPart) and self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __repr__(self) -> str:
        extra = "" if self.grid_index is None else f", grid_index=<{len(self.grid_index)} pts>"
        return f"ReadPart({self.path!r}, slices={self.slices}, squeeze={self.squeeze}{extra})"


class ReadBuffer:
    """Resolves original :class:`ReadPart` requests against factorized arrays.

    After :func:`factorize`, several original parts may have been merged into
    one bounding-box part.  ``ReadBuffer.__getitem__`` extracts the correct
    sub-rows for each original request.

    Parameters
    ----------
    raw : dict
        Maps each factorized (merged) ``ReadPart`` to its read array.
    mapping : dict
        Maps each original ``ReadPart`` to ``(merged_part, row_offset, grid_cols)``.
        ``row_offset`` is the index of the first date row of the original part
        within the merged read.  ``grid_cols`` is ``None`` (the merged grid axis
        is exactly this part's) or a list of column positions selecting this
        part's grid points out of the merged grid-index *union*.
    """

    def __init__(
        self,
        raw: dict[ReadPart, NDArray],
        mapping: dict[ReadPart, tuple[ReadPart, int, Any]],
    ) -> None:
        self._raw = raw
        self._mapping = mapping

    def __getitem__(self, part: ReadPart) -> NDArray:
        merged, offset, grid_cols = self._mapping.get(part, (part, 0, None))
        data = self._raw[merged]
        s, e, t = part.slices[0]
        n_rows = len(range(s, e, t))
        data = data[offset : offset + n_rows]
        if grid_cols is not None:
            data = data[..., grid_cols]
        return data


def factorize(
    parts: list[ReadPart],
) -> tuple[list[ReadPart], dict[ReadPart, tuple[ReadPart, int, Any]]]:
    """Merge parts that share a zarr array and non-date dimensions.

    Groups parts by ``(id(data), slices[1:])``.  Within each group the date
    slices are merged into a single bounding-box slice
    ``slice(min_start, max_stop, step)``.  Parts that carry a ``grid_index``
    (grid-subset pushdown) are additionally **unioned on the grid axis**: all
    grid points any part of the group needs are read once, and each original
    part is mapped to its columns within that union.  This is what lets a store
    shared by several cutouts (e.g. one global model in a ``multi=``) be read a
    single time even when each cutout asks for a different grid subset.

    Returns
    -------
    merged_parts : list[ReadPart]
        Deduplicated, merged parts ready for :func:`execute_parts`.
    mapping : dict
        ``{original_part: (merged_part, row_offset, grid_cols)}``.  ``row_offset``
        is the number of rows from the merged read's start to the part's date
        slice; ``grid_cols`` is ``None`` or the part's columns within the merged
        grid-index union.
    """
    # Deduplicate while preserving order
    seen: dict[ReadPart, None] = {}
    for p in parts:
        seen.setdefault(p, None)
    unique = list(seen)

    # Group by zarr array identity + non-date axes.  Parts with a grid_index are
    # grouped together (regardless of which grid points) so their grid indices
    # can be unioned; parts without are kept separate (full grid axis as-is).
    groups: dict[tuple, list[ReadPart]] = defaultdict(list)
    for p in unique:
        kind = "grid" if p.grid_index is not None else None
        key = (id(p.data), p.slices[1:], kind)
        groups[key].append(p)

    merged_parts: list[ReadPart] = []
    mapping: dict[ReadPart, tuple[ReadPart, int, Any]] = {}

    for (data_id, rest_slices, kind), group in groups.items():  # noqa: B007  (data_id unused intentionally)
        group.sort(key=lambda p: p.slices[0][0])

        merged_start = group[0].slices[0][0]
        merged_stop = max(p.slices[0][1] for p in group)
        merged_step = group[0].slices[0][2]
        merged_date: NormSlice = (merged_start, merged_stop, merged_step)

        if kind == "grid":
            union = sorted({g for p in group for g in p.grid_index})
            position = {g: i for i, g in enumerate(union)}
            merged_grid_index: tuple[int, ...] | None = tuple(union)
        else:
            position = {}
            merged_grid_index = None

        merged = ReadPart(
            path=group[0].path,
            data=group[0].data,
            slices=(merged_date,) + rest_slices,
            squeeze=(),
            grid_index=merged_grid_index,
        )
        merged_parts.append(merged)

        for p in group:
            # How many rows from merged_start to p's start, counting by step
            offset = len(range(merged_start, p.slices[0][0], merged_step))
            grid_cols = [position[g] for g in p.grid_index] if kind == "grid" else None
            mapping[p] = (merged, offset, grid_cols)

    if READ_PARTS_DEBUG:
        LOG.debug("factorize: %d → %d part(s)", len(unique), len(merged_parts))
        for mp in merged_parts:
            LOG.debug("  merged: %s  slices=%s", mp.path, mp.slices)

    return merged_parts, mapping


def gather_parts(children: "Any") -> "list | None":
    """Flatten child ``collect_read_parts`` results, propagating unsupported.

    Multi-dataset wrappers (Concat, Join, Merge, Grids, Cutout, Multi) collect
    from several children.  If **any** child returns ``None`` (it does not support
    two-step for this index), the whole wrapper returns ``None`` so the caller
    falls back to eager — no exception.

    Parameters
    ----------
    children : iterable
        Iterable of child ``collect_read_parts`` results (each ``list`` or ``None``).
    """
    out: list = []
    for child in children:
        if child is None:
            return None
        out.extend(child)
    return out


def execute_parts(parts: list[ReadPart]) -> dict[ReadPart, NDArray]:
    """Execute all reads sequentially and return a buffer of results.

    Reads are run one after another.  Parallelising *across* parts with Python
    threads does not help: chunk decompression (blosc) is CPU/GIL-bound, so a
    thread pool only adds contention and is measurably slower for the common
    few-large-chunks case (e.g. a cutout reading a LAM chunk + a global chunk).
    Concurrency where it actually helps — overlapping S3 latency across the
    chunks of a single array — already happens inside zarr/anemoi ``getitems``.

    Parameters
    ----------
    parts : list[ReadPart]
        Factorized parts from :func:`factorize` (no duplicates).

    Returns
    -------
    dict
        Maps each part to the numpy array returned by the zarr read.
    """
    buffer: dict[ReadPart, NDArray] = {}
    for part in parts:
        if READ_PARTS_DEBUG:
            LOG.debug("execute: %s  slices=%s", part.path, part.slices)
        buffer[part] = part.execute()
    return buffer


def two_step_read(dataset: "Dataset", index: "FullIndex") -> "NDArray | None":
    """Execute ``dataset[index]`` via collect → factorize → execute → reconstruct.

    Main entry point for the two-step path.

    Fallback is signalled by **return value, not exception** (reading via the
    eager path is a normal, expected outcome — many wrappers do not implement
    two-step).  Returns ``None`` when this index cannot be served by the fast
    path; the caller (the gated ``__getitem__``) then uses the eager reader.
    ``None`` happens when:

    * a wrapper's :meth:`~Dataset.collect_read_parts` returns ``None`` (it does
      not support two-step for this index), or
    * the index is not expressible as rectangular slices (e.g. a list/array on a
      non-grid axis) — :func:`index_to_slices` raises ``ValueError``/``TypeError``/
      ``AttributeError``, caught here.

    Genuine errors (``IndexError`` for out-of-range integers) propagate.

    Parameters
    ----------
    dataset : Dataset
        The root dataset (may be a wrapper chain).
    index : FullIndex
        The index passed to ``__getitem__``.

    Returns
    -------
    NDArray or None
        The result (same as the eager path), or ``None`` to request fallback.
    """
    try:
        parts = dataset.collect_read_parts(index)
    except (ValueError, TypeError, AttributeError):
        return None  # index not normalisable to rectangular slices → eager

    if parts is None:
        return None  # wrapper does not support two-step for this index → eager

    if READ_PARTS_DEBUG:
        LOG.debug(
            "two_step_read: %s[%r] → %d part(s)",
            type(dataset).__name__,
            index,
            len(parts),
        )
        for p in parts:
            LOG.debug("  %r", p)

    factorized, part_map = factorize(parts)

    if READ_PARTS_DEBUG:
        LOG.debug("factorized → %d part(s)", len(factorized))

    raw = execute_parts(factorized)
    buffer = ReadBuffer(raw=raw, mapping=part_map)

    return dataset.read_from_buffer(index, buffer)
