# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from typing import Union

LOG = logging.getLogger(__name__)

ALL = object()


class ChunkFilter:
    """A filter to determine which chunks to process based on the specified parts.

    Attributes
    ----------
    total : int
        The total number of chunks.
    allowed : object or list
        The chunks that are allowed to be processed.
    """

    def __init__(self, *, parts: Union[str, list], total: int):
        """Initializes the ChunkFilter with the given parts and total number of chunks.

        Parameters
        ----------
        parts : str or list
            The parts to process, specified as 'i/n' or a list of such strings.
        total : int
            The total number of chunks.

        Raises
        ------
        ValueError
            If the parts format is invalid.
        AssertionError
            If the chunk number is invalid.
        Warning
            If the number of chunks is larger than the total number of chunks.
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

        LOG.debug(f"Running parts: {parts}")

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

        if self.allowed == ALL:
            return True
        return i in self.allowed

    def __iter__(self) -> iter:
        """Iterates over the allowed chunks.

        Yields
        ------
        int
            The next allowed chunk number.
        """
        for i in range(self.total):
            if self(i):
                yield i

    def __len__(self) -> int:
        """Returns the number of allowed chunks.

        Returns
        -------
        int
            The number of allowed chunks.
        """
        return len([_ for _ in self])
