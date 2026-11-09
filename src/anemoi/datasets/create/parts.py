# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings

LOG = logging.getLogger(__name__)

ALL = object()


def part_bounds(parts: str | list | None, total: int) -> tuple[int, int, int]:
    """Return ``(part_index, lo, hi)`` for the contiguous slice ``[lo, hi)`` of ``[0, total)``.

    Mirrors :class:`PartFilter`'s 1-based ``i/n`` contiguous split. The bounds are derived from
    ``total`` so that empty parts (``n`` larger than ``total``) still yield ``lo == hi`` at the
    correct boundary, keeping the per-part results contiguous when they are merged.

    Parameters
    ----------
    parts : str or list or None
        The part to select, as ``'i/n'`` (1-based) or a one-element list of such a string.
        ``None``, ``'all'`` or ``'*'`` select the whole range.
    total : int
        The number of items to split.

    Returns
    -------
    tuple[int, int, int]
        The 0-based part index and the bounds ``lo`` and ``hi``.
    """
    if isinstance(parts, list):
        if len(parts) == 0:
            parts = None
        elif len(parts) == 1:
            parts = parts[0]
        else:
            raise ValueError(f"Invalid parts format: {parts}. Must be a single 'i/n'.")

    if not parts or parts.lower() in ("all", "*"):
        return 0, 0, total

    i, n = (int(x) for x in parts.split("/"))
    assert 0 < i <= n, f"Invalid part {parts}: must be in the form 'i/n' with 1 <= i <= n."

    chunk_size = total / n
    lo = sum(1 for x in range(total) if x < (i - 1) * chunk_size)
    hi = sum(1 for x in range(total) if x < i * chunk_size)
    return i - 1, lo, hi


class PartFilter:
    """A filter to determine which parts to process based on the specified parts.

    Attributes
    ----------
    total : int
        The total number of parts.
    allowed : object or list
        The parts that are allowed to be processed.
    """

    def __init__(self, *, parts: str | list, total: int):
        """Initializes the PartFilter with the given parts and total number of parts.

        Parameters
        ----------
        parts : str or list
            The parts to process, specified as 'i/n' or a list of such strings.
        total : int
            The total number of parts.

        Raises
        ------
        ValueError
            If the parts format is invalid.
        AssertionError
            If the chunk number is invalid.
        Warning
            If the number of parts is larger than the total number of parts.
        """
        self.total = total

        if isinstance(parts, list):
            if len(parts) == 1:
                parts = parts[0]
            elif len(parts) == 0:
                parts = None
            else:
                raise ValueError(f"Invalid parts format: {parts}. Must be in the form 'i/n'.")

        if not parts:
            parts = "all"

        assert isinstance(parts, str), f"Argument parts must be a string, got {parts}."

        if parts.lower() == "all" or parts == "*":
            self.allowed = ALL
            return

        assert "/" in parts, f"Invalid parts format: {parts}. Must be in the form 'i/n'."

        i, n = parts.split("/")
        i, n = int(i), int(n)

        assert i > 0, f"Chunk number {i} must be positive."
        assert i <= n, f"Chunk number {i} must be less than total chunks {n}."
        if n > total:
            warnings.warn(
                f"Number of chunks {n} is larger than the total number of chunks: {total}. "
                "Some chunks will be empty."
            )

        chunk_size = total / n
        parts = [x for x in range(total) if x >= (i - 1) * chunk_size and x < i * chunk_size]

        for i in parts:
            if i < 0 or i >= total:
                raise AssertionError(f"Invalid chunk number {i}. Must be between 0 and {total - 1}.")
        if not parts:
            warnings.warn(f"Nothing to do for chunk {i}/{n}.")

        LOG.info(f"Running parts: {parts}")

        self.allowed = parts

    def __call__(self, i: int) -> bool:
        """Checks if the given chunk number is allowed to be processed.

        Parameters
        ----------
        i : int
            The chunk number to check.

        Returns
        -------
        bool
            True if the chunk is allowed, False otherwise.

        Raises
        ------
        AssertionError
            If the chunk number is invalid.
        """
        if i < 0 or i >= self.total:
            raise AssertionError(f"Invalid chunk number {i}. Must be between 0 and {self.total - 1}.")

        return (self.allowed is ALL) or (i in self.allowed)

    # def __iter__(self) -> iter:
    #     """Iterates over the allowed chunks.

    #     Yields
    #     ------
    #     int
    #         The next allowed chunk number.
    #     """
    #     for i in range(self.total):
    #         if self(i):
    #             yield i

    # def __len__(self) -> int:
    #     """Returns the number of allowed chunks.

    #     Returns
    #     -------
    #     int
    #         The number of allowed chunks.
    #     """
    #     return len([_ for _ in self])
