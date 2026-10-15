# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A covering partitions a window into subwindows, each rebuilt from archived fields.

A :class:`Subwindow` separates two questions that a flat list of signed intervals
answers at once, and therefore answers badly:

- *which part of the window is this?* -- ``interval``, always forward;
- *how do I rebuild its value from what the archive holds?* -- ``contributions``,
  signed, summed.

For an archive storing per-step values a subwindow is **direct**: one positive
contribution equal to the interval. For an archive storing values accumulated from the
start of the forecast it is **differenced**: ``+a(0, end) - a(0, start)``. Both are
subwindows, so a consumer no longer has to know which it is looking at.

That distinction is invisible in a flat list, and it only *stays* invisible because
addition is associative -- the grouping cannot change a sum. For any other reduction
the grouping is precisely the information needed.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from anemoi.datasets.create.intervals import SignedInterval


@dataclass(frozen=True)
class Subwindow:
    """One part of a window, and how to rebuild its value.

    Parameters
    ----------
    interval : SignedInterval
        The part itself. Always forward (``start < end``).
    contributions : tuple of SignedInterval
        The archived intervals whose **signed sum** reconstructs ``interval``. A
        single positive contribution equal to ``interval`` means the archive holds
        the part outright.
    """

    interval: SignedInterval
    contributions: tuple[SignedInterval, ...]

    @property
    def is_direct(self) -> bool:
        """Whether the archive holds this part outright, with no reconstruction."""
        return len(self.contributions) == 1 and self.contributions[0].sign > 0

    @property
    def weight(self) -> float:
        """How much this part counts for in a weighted reduction: its own length.

        A sample weighs 1 instead; keeping the weight on the *part* is what lets one
        ``Mean`` serve both, length-weighted where the partition mixes granularities.
        """
        return abs(self.interval.length)

    @property
    def start(self) -> datetime.datetime:
        return self.interval.start

    @property
    def end(self) -> datetime.datetime:
        return self.interval.end

    def __repr__(self) -> str:
        how = "direct" if self.is_direct else f"{len(self.contributions)} contributions"
        return f"Subwindow({self.interval.start:%Y%m%d.%H%M} -> {self.interval.end:%Y%m%d.%H%M}, {how})"


def direct(interval: SignedInterval) -> Subwindow:
    """Build a subwindow the archive holds outright."""
    return Subwindow(interval=interval, contributions=(interval,))


def contributions_of(subwindows) -> list[SignedInterval]:
    """Flatten subwindows back to the archived intervals that must be retrieved."""
    return [c for subwindow in subwindows for c in subwindow.contributions]


def validate_contributions(subwindow: Subwindow) -> None:
    """Raise unless a subwindow's contributions reconstruct its own interval.

    The signed lengths must sum to the subwindow's length. Cheap, and it catches a
    covering that groups its contributions wrongly.

    Parameters
    ----------
    subwindow : Subwindow
        The subwindow to check.

    Raises
    ------
    ValueError
        If the contributions do not reconstruct ``subwindow.interval``.
    """
    if not subwindow.contributions:
        raise ValueError(f"{subwindow} has no contributions.")

    total = sum(c.length for c in subwindow.contributions)
    if total != subwindow.interval.length:
        detail = "\n".join(f"    {'+' if c.length >= 0 else '-'} {c}" for c in subwindow.contributions)
        raise ValueError(
            f"{subwindow}: its contributions' signed lengths total {total}s but the subwindow "
            f"spans {subwindow.interval.length}s, so they do not reconstruct it. They were:\n{detail}"
        )


def validate_partition(subwindows, start: datetime.datetime, end: datetime.datetime) -> None:
    """Raise unless ``subwindows`` partitions ``[start, end]`` exactly.

    Forward-only, gapless, no overlap and no overhang -- the mathematical sense of a
    partition. This is the window-level invariant of a covering and it holds whatever
    operation will reduce the subwindows: the signed freedom lives *inside* a
    subwindow, never between them.

    Parameters
    ----------
    subwindows : list of Subwindow
        The covering to check.
    start, end : datetime.datetime
        The window it must reproduce exactly.

    Raises
    ------
    ValueError
        If the covering is empty, runs backwards, leaves a gap, overlaps, or extends
        outside ``[start, end]``.
    """
    subwindows = list(subwindows)
    if not subwindows:
        raise ValueError(f"Empty covering for window {start} -> {end}.")

    backwards = [s for s in subwindows if s.interval.length <= 0]
    if backwards:
        raise ValueError(f"Covering of {start} -> {end} contains subwindow(s) that do not run forwards: {backwards}.")

    ordered = sorted(subwindows, key=lambda s: s.start)
    if ordered[0].start != start or ordered[-1].end != end:
        raise ValueError(
            f"Covering of {start} -> {end} does not line up with the window: "
            f"it spans {ordered[0].start} -> {ordered[-1].end}."
        )
    for previous, following in zip(ordered, ordered[1:]):
        if previous.end != following.start:
            what = "gap" if previous.end < following.start else "overlap"
            raise ValueError(f"Covering of {start} -> {end} has a {what} between {previous} and {following}.")


def group_intervals_into_subwindows(intervals, start: datetime.datetime, end: datetime.datetime) -> list[Subwindow]:
    """Group a chain of signed intervals into the subwindows it reconstructs.

    Every covering produces a *chain*: each interval starts where the previous one
    ended, running from ``start`` to ``end``. For a per-step archive each link advances
    the window by one part, so each is its own subwindow. For an archive accumulated
    from the start of the forecast the chain dips *outside* the window -- ``-a(0,6)``
    then ``+a(0,12)`` reach ``[6,12]`` by way of the basetime -- and those two links are
    one reconstructed part, not two.

    So a subwindow closes only where the walk stands on a time that advances the
    partition: inside the window, and beyond the current subwindow's start. Everywhere
    else the walk is mid-reconstruction.

    Parameters
    ----------
    intervals : list of SignedInterval
        The chain produced by a covering.
    start, end : datetime.datetime
        The window being covered.

    Returns
    -------
    list of Subwindow
        The subwindows partitioning ``[start, end]``.

    Raises
    ------
    ValueError
        If the chain does not end on a partition boundary, or the subwindows do not
        partition the window.
    """
    subwindows: list[Subwindow] = []
    pending: list[SignedInterval] = []
    subwindow_start = start

    for link in intervals:
        pending.append(link)
        position = link.end
        if subwindow_start < position <= end:
            if len(pending) == 1:
                # the archive holds this part outright; keep the interval's own base
                subwindow = direct(pending[0])
            else:
                subwindow = Subwindow(
                    interval=SignedInterval(start=subwindow_start, end=position),
                    contributions=tuple(pending),
                )
            validate_contributions(subwindow)
            subwindows.append(subwindow)
            pending = []
            subwindow_start = position

    if pending:
        raise ValueError(
            f"Covering of {start} -> {end} ended mid-reconstruction, with "
            f"{len(pending)} interval(s) not closing a subwindow: {pending}."
        )

    validate_partition(subwindows, start, end)
    return subwindows
