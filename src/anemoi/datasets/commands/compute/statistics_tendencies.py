# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Standalone tendency statistics for the ``compute`` command.

A *tendency* is the difference ``value(t) - value(t - delta)``, where ``delta`` is
a whole number of time steps. Statistics are accumulated over those differences
using the same numerically-stable accumulator as plain statistics. A sliding
window of the last ``delta`` rows is carried across chunks so that tendencies
straddling a chunk boundary are still computed; this works because the loop is
single-process and strictly chronological.
"""

import datetime
import logging
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .statistics import DEFAULT_CHUNK_SIZE
from .statistics import Accumulator
from .statistics import _progress
from .statistics import iter_chunks

LOG = logging.getLogger(__name__)

_UNITS = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_delta(value: Any) -> datetime.timedelta:
    """Parse a time delta into a :class:`datetime.timedelta` (standalone).

    Accepts a ``timedelta``, a bare integer (interpreted as hours), or a string
    such as ``"6h"``, ``"30m"``, ``"1d"`` or ``"3600s"``. Kept self-contained so
    the ``compute`` command does not depend on the rest of anemoi.

    Parameters
    ----------
    value : timedelta, int or str
        The delta to parse.

    Returns
    -------
    datetime.timedelta
        The parsed delta.
    """
    if isinstance(value, datetime.timedelta):
        return value
    if isinstance(value, bool):  # avoid True/False sneaking in as ints
        raise ValueError(f"Invalid delta: {value!r}")
    if isinstance(value, int):
        return datetime.timedelta(hours=value)
    s = str(value).strip().lower()
    if s.isdigit():
        return datetime.timedelta(hours=int(s))
    match = re.fullmatch(r"(\d+)\s*([smhdw])", s)
    if not match:
        raise ValueError(f"Cannot parse delta {value!r}; use e.g. '6h', '30m', '1d'")
    return datetime.timedelta(**{_UNITS[match.group(2)]: int(match.group(1))})


def delta_to_steps(delta: Any, frequency: Any) -> int:
    """Convert a time delta into a whole number of dataset time steps.

    Parameters
    ----------
    delta : str, int or timedelta
        The tendency delta (e.g. ``"6h"``, ``6`` or a ``timedelta``).
    frequency : timedelta
        The dataset frequency.

    Returns
    -------
    int
        The delta expressed as a number of time steps.
    """
    td = parse_delta(delta)
    ratio = td / parse_delta(frequency)
    if int(ratio) != ratio:
        raise ValueError(f"Tendency delta {delta} ({td}) is not a multiple of dataset frequency {frequency}")
    steps = int(ratio)
    if steps < 1:
        raise ValueError(f"Tendency delta {delta} must be at least one time step")
    return steps


class TendencyAccumulator:
    """Accumulate statistics of temporal tendencies with a sliding window.

    Parameters
    ----------
    variables : list of str
        Names of the variables.
    delta : int
        The tendency in number of time steps (``value(t) - value(t - delta)``).
    allow_nans : bool, optional
        Passed through to the underlying accumulator.
    """

    def __init__(self, variables: list[str], delta: int, allow_nans: bool = False) -> None:
        self.variables = list(variables)
        self.delta = int(delta)
        self.allow_nans = bool(allow_nans)
        self._accumulator = Accumulator(variables, allow_nans=allow_nans)
        self._window: NDArray[Any] | None = None

    def seed_window(self, data: NDArray[Any]) -> None:
        """Seed the sliding window with the rows preceding a segment.

        Used by the parallel engine so that tendencies straddling a segment
        boundary are still computed. Only the last ``delta`` rows are kept and
        nothing is accumulated.

        Parameters
        ----------
        data : ndarray
            The rows immediately before the segment start (time on axis 0).
        """
        data = np.asarray(data, dtype=np.float64)
        if len(data) == 0:
            self._window = None
        else:
            self._window = np.array(data[-self.delta :], copy=True)

    def merge(self, other: "TendencyAccumulator") -> "TendencyAccumulator":
        """Merge another tendency accumulator into a new one.

        Only the underlying tendency statistics are merged; sliding windows are
        transient and are not carried across a merge.

        Parameters
        ----------
        other : TendencyAccumulator
            The accumulator to merge with; must share variables and ``delta``.

        Returns
        -------
        TendencyAccumulator
            A new merged accumulator.
        """
        if self.delta != other.delta:
            raise ValueError("Cannot merge tendency accumulators with different deltas")
        result = TendencyAccumulator(self.variables, self.delta, allow_nans=self.allow_nans or other.allow_nans)
        result._accumulator = self._accumulator.merge(other._accumulator)
        return result

    def update(self, data: NDArray[Any]) -> None:
        """Accumulate tendencies for a chronologically-ordered batch of data.

        Parameters
        ----------
        data : ndarray
            Array whose axis 0 is time and axis 1 is the variable axis.
        """
        data = np.asarray(data, dtype=np.float64)
        if len(data) == 0:
            return

        combined = data if self._window is None else np.concatenate([self._window, data], axis=0)

        if len(combined) > self.delta:
            tendencies = combined[self.delta :] - combined[: -self.delta]
            self._accumulator.update(tendencies)

        # Keep the last `delta` rows for the next chunk.
        self._window = np.array(combined[-self.delta :], copy=True)

    def statistics(self) -> dict[str, NDArray[np.float64]]:
        """Return the tendency statistics (see :meth:`Accumulator.statistics`)."""
        return self._accumulator.statistics()


def compute_tendency_statistics(
    dataset: Any,
    delta: int,
    start: int = 0,
    end: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    allow_nans: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Compute tendency statistics over a dataset with a simple chunked loop.

    Parameters
    ----------
    dataset : Dataset
        An opened anemoi dataset.
    delta : int
        The tendency in number of time steps.
    start : int, optional
        First time index to include.
    end : int, optional
        One past the last time index to include.
    chunk_size : int, optional
        Number of time steps read per chunk.
    allow_nans : bool, optional
        Whether to ignore NaNs per-variable.

    Returns
    -------
    dict
        The tendency statistics.
    """
    accumulator = TendencyAccumulator(list(dataset.variables), delta, allow_nans=allow_nans)
    for lo, hi in _progress(iter_chunks(len(dataset), start, end, chunk_size)):
        accumulator.update(dataset[lo:hi])
    return accumulator.statistics()
