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
  3. :func:`execute_parts` — sequential zarr reads (parallelism: Phase 5).
  4. ``ds.read_from_cache(n, cache)`` — reassembles the result from cached
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
    """

    __slots__ = ("path", "data", "slices", "squeeze")

    def __init__(
        self,
        path: str,
        data: Any,
        slices: tuple[NormSlice, ...],
        squeeze: tuple[int, ...],
    ) -> None:
        self.path = path
        self.data = data
        self.slices = slices
        self.squeeze = squeeze

    @classmethod
    def from_raw_slices(
        cls,
        path: str,
        data: Any,
        slices: tuple[slice, ...],
        squeeze: tuple[int, ...],
    ) -> "ReadPart":
        """Build from a tuple of concrete ``slice`` objects."""
        norm: tuple[NormSlice, ...] = tuple(
            (s.start, s.stop, s.step) for s in slices
        )
        return cls(path=path, data=data, slices=norm, squeeze=squeeze)

    def to_zarr_index(self) -> tuple[slice, ...]:
        """Convert stored slices back to zarr-compatible ``slice`` objects."""
        return tuple(slice(s, e, t) for s, e, t in self.slices)

    def execute(self) -> NDArray:
        """Execute the zarr read and return the array."""
        return self.data[self.to_zarr_index()]

    def _identity(self) -> tuple:
        return (id(self.data), self.slices)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReadPart) and self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __repr__(self) -> str:
        return f"ReadPart({self.path!r}, slices={self.slices}, squeeze={self.squeeze})"


class ReadCache:
    """Resolves original :class:`ReadPart` requests against factorized arrays.

    After :func:`factorize`, several original parts may have been merged into
    one bounding-box part.  ``ReadCache.__getitem__`` extracts the correct
    sub-rows for each original request.

    Parameters
    ----------
    raw : dict
        Maps each factorized (merged) ``ReadPart`` to its read array.
    mapping : dict
        Maps each original ``ReadPart`` to ``(merged_part, row_offset)``.
        The row offset is the index of the first date row of the original
        part within the merged read.
    """

    def __init__(
        self,
        raw: dict[ReadPart, NDArray],
        mapping: dict[ReadPart, tuple[ReadPart, int]],
    ) -> None:
        self._raw = raw
        self._mapping = mapping

    def __getitem__(self, part: ReadPart) -> NDArray:
        merged, offset = self._mapping.get(part, (part, 0))
        data = self._raw[merged]
        s, e, t = part.slices[0]
        n_rows = len(range(s, e, t))
        return data[offset : offset + n_rows]


def factorize(
    parts: list[ReadPart],
) -> tuple[list[ReadPart], dict[ReadPart, tuple[ReadPart, int]]]:
    """Merge parts that share a zarr array and non-date dimensions.

    Groups parts by ``(id(data), slices[1:])``.  Within each group the date
    slices are merged into a single bounding-box slice
    ``slice(min_start, max_stop, step)``.

    Returns
    -------
    merged_parts : list[ReadPart]
        Deduplicated, merged parts ready for :func:`execute_parts`.
    mapping : dict
        ``{original_part: (merged_part, row_offset)}``.  The offset is the
        number of rows from the start of the merged read to the start of the
        original part's date slice.
    """
    # Deduplicate while preserving order
    seen: dict[ReadPart, None] = {}
    for p in parts:
        seen.setdefault(p, None)
    unique = list(seen)

    # Group by zarr array identity + non-date axes
    groups: dict[tuple, list[ReadPart]] = defaultdict(list)
    for p in unique:
        key = (id(p.data), p.slices[1:])
        groups[key].append(p)

    merged_parts: list[ReadPart] = []
    mapping: dict[ReadPart, tuple[ReadPart, int]] = {}

    for (data_id, rest_slices), group in groups.items():  # noqa: B007  (data_id unused intentionally)
        group.sort(key=lambda p: p.slices[0][0])

        merged_start = group[0].slices[0][0]
        merged_stop = max(p.slices[0][1] for p in group)
        merged_step = group[0].slices[0][2]

        merged_date: NormSlice = (merged_start, merged_stop, merged_step)
        merged = ReadPart(
            path=group[0].path,
            data=group[0].data,
            slices=(merged_date,) + rest_slices,
            squeeze=(),
        )
        merged_parts.append(merged)

        for p in group:
            # How many rows from merged_start to p's start, counting by step
            offset = len(range(merged_start, p.slices[0][0], merged_step))
            mapping[p] = (merged, offset)

    if READ_PARTS_DEBUG:
        LOG.debug("factorize: %d → %d part(s)", len(unique), len(merged_parts))
        for mp in merged_parts:
            LOG.debug("  merged: %s  slices=%s", mp.path, mp.slices)

    return merged_parts, mapping


def execute_parts(parts: list[ReadPart]) -> dict[ReadPart, NDArray]:
    """Execute all reads sequentially and return a cache of results.

    Parameters
    ----------
    parts : list[ReadPart]
        Factorized parts from :func:`factorize` (no duplicates).

    Returns
    -------
    dict
        Maps each part to the numpy array returned by the zarr read.
    """
    cache: dict[ReadPart, NDArray] = {}
    for part in parts:
        if READ_PARTS_DEBUG:
            LOG.debug("execute: %s  slices=%s", part.path, part.slices)
        cache[part] = part.execute()
    return cache


def two_step_read(dataset: "Dataset", index: "FullIndex") -> NDArray:
    """Execute ``dataset[index]`` via collect → factorize → execute → reconstruct.

    This is the main entry point for the two-step read path.  All wrapper
    classes in the dataset chain must implement :meth:`~Dataset.collect_read_parts`
    and :meth:`~Dataset.read_from_cache` for this to work end-to-end.

    Parameters
    ----------
    dataset : Dataset
        The root dataset (may be a wrapper chain).
    index : FullIndex
        The index passed to ``__getitem__``.

    Returns
    -------
    NDArray
        Same result as ``dataset[index]`` via the legacy path.
    """
    parts = dataset.collect_read_parts(index)

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
    cache = ReadCache(raw=raw, mapping=part_map)

    return dataset.read_from_cache(index, cache)
