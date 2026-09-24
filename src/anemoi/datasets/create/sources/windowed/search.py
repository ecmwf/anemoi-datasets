# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Find the archived intervals that cover a window.

A Dijkstra over a graph whose nodes are ``(time, base, covered)`` and whose edges are
the intervals the archive offers at that time -- forwards, or the negation of one that
ends there, which is how a window is reached from a cumulative archive by way of the
basetime.

The walk returns a *chain*: each interval starts where the last one ended. What that
chain reconstructs is recovered afterwards by
:func:`..subwindows.group_intervals_into_subwindows`.
"""

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from heapq import heappop
from heapq import heappush

from anemoi.datasets.create.intervals import SignedInterval

LOG = logging.getLogger(__name__)


@dataclass(order=True)
class HeapState:
    total_cost: float
    covered: float
    counter: int
    current_time: datetime
    current_base: datetime | None
    path: list[SignedInterval] = field(compare=False)


def search_intervals(
    start: datetime,
    end: datetime,
    candidates: Callable,
    /,
    switch_penalty: int = 24 * 3600 * 7,
    max_delta: timedelta = timedelta(hours=24 * 2),
    error_on_fail: bool = True,
) -> list[SignedInterval] | None:
    """Find a path of intervals covering [start, end] with minimal base switches, then minimal total absolute length.
    Uses a Dijkstra-like algorithm to find the optimal path.

    Args:
        start: Start datetime of the target interval.

        end: End datetime of the target interval.

        candidates: A function(current: datetime, current_base: Optional[datetime], start: datetime, end: datetime) -> Iterable[SignedInterval]
            that provides candidate intervals covering the current time.

        switch_penalty: Penalty (in seconds) for switching bases between intervals.

        max_delta: Maximum allowed deviation from start/end for search.

        error_on_fail: Whether to raise an error if coverage cannot be found.

    Returns:
        A list of SignedInterval objects covering [start, end], or None if no coverage found and error_on_fail is False.

    """
    target_length = (end - start).total_seconds()

    # Both ends of the window must be boundaries the archive offers, and this is
    # decidable before searching. Every edge sets `current_time = interval.end` and adds
    # the same `interval.length` to `covered`, and every candidate starts at
    # `current_time` -- so `covered == current_time - start` at every reachable state,
    # and the goal `covered == target_length` is reached exactly when
    # `current_time == end`. The last edge of any successful walk therefore lands on
    # `end`, so if nothing starts or ends there, no route exists however long we look.
    #
    # A description with recurring base dates generates runs for ever, so such a search
    # does not fail, it *wanders*: there is always another edge, and the walk runs on
    # until the state budget trips -- weeks past a window that was unreachable from the
    # first call. Knowing the window is doomed lets us cap that cheaply, while still
    # searching far enough for the message to say how close the archive gets.
    offered_at_start = list(candidates(start))
    offered_at_end = offered_at_start if end == start else list(candidates(end))
    budget = 200 if (not offered_at_start or not offered_at_end) else 1000

    pq: list[HeapState] = []  # pq: priority queue
    counter = itertools.count()
    heappush(
        pq,
        HeapState(total_cost=0.0, covered=0.0, counter=next(counter), current_time=start, current_base=None, path=[]),
    )

    visited: dict[tuple[datetime, datetime | None, float], float] = {}

    # Kept for the failure message: a search that gets nowhere and one that gets
    # almost there are different problems, and the old message could not tell them
    # apart. `closest` is the state that came nearest to covering the window.
    closest: HeapState | None = None

    while pq:
        state = heappop(pq)
        key = (state.current_time, state.current_base, state.covered)

        if key in visited and state.total_cost >= visited[key]:
            continue
        visited[key] = state.total_cost

        # Goal: cumulative coverage matches target
        if state.covered == target_length:
            return state.path

        if (len(visited) > budget) and (state.current_time > end + max_delta or state.current_time < start - max_delta):
            # Same failure as running out of candidates, reached a different way, so it
            # gets the same explanation rather than a bare state count.
            msg = _no_coverage(
                start,
                end,
                target_length,
                offered_at_start,
                offered_at_end,
                closest,
                gave_up=(len(visited), state.current_time),
            )
            if error_on_fail:
                raise ValueError(msg)
            LOG.warning(msg)
            return None

        if closest is None or abs(target_length - state.covered) < abs(target_length - closest.covered):
            closest = state

        offered = offered_at_start if state.current_time == start else list(candidates(state.current_time))

        for interval in offered:
            if interval.start != state.current_time:
                raise ValueError(
                    f"Candidate interval {interval} does not start or end at current_time {state.current_time}"
                )

            # Edge cost = abs(length) + switch penalty if base changes
            edge_cost = abs(interval.length)
            if state.current_base is not None and state.current_base != interval.base:
                edge_cost += switch_penalty

            heappush(
                pq,
                HeapState(
                    total_cost=state.total_cost + edge_cost,
                    covered=state.covered + interval.length,
                    counter=next(counter),  # counter only used to break ties in heapq
                    current_time=interval.end,
                    current_base=interval.base,
                    path=state.path + [interval],
                ),
            )

    msg = _no_coverage(start, end, target_length, offered_at_start, offered_at_end, closest)
    if error_on_fail:
        raise ValueError(msg)
    LOG.warning(msg)
    return None


def _no_coverage(
    start: datetime,
    end: datetime,
    target_length: float,
    offered_at_start: list[SignedInterval],
    offered_at_end: list[SignedInterval],
    closest: "HeapState | None",
    gave_up: tuple[int, datetime] | None = None,
) -> str:
    """Explain a search that could not close the window.

    Four things go wrong and they want different fixes: the window's start is not a
    boundary the archive offers, so it cannot even be entered; its *end* is not one, so
    nothing can close it however the walk goes; something is offered at both but no
    combination connects them; or a route got most of the way and stalled.

    The end case is worth naming explicitly. Every edge advances ``current_time`` and
    ``covered`` by the same amount, so ``covered == current_time - start`` throughout and
    the goal is reached exactly when ``current_time == end`` -- the last edge of any
    successful walk lands on ``end``. If nothing starts or ends there, no route exists.
    """
    lines = [f"Cannot find coverage of {start} → {end}"]

    if not offered_at_start:
        lines.append(
            f"  Nothing in the archive description starts or ends at {start}, so the window "
            "cannot even be entered. Its start has to be a step boundary the source data "
            "actually offers."
        )
        return "\n".join(lines)

    if not offered_at_end:
        lines.append(
            f"  Nothing in the archive description starts or ends at {end}, so nothing can "
            "close the window: every covering ends on its last archived interval, and none "
            "of them ends there."
        )

    lines.append(f"  Starting at {start}, the description offers {len(offered_at_start)} interval(s):")
    for interval in offered_at_start[:8]:
        lines.append(f"    {interval}")
    if len(offered_at_start) > 8:
        lines.append(f"    ... and {len(offered_at_start) - 8} more")

    if closest is not None and closest.path:
        covered = timedelta(seconds=closest.covered)
        if closest.covered > target_length:
            how = f"overshot it by {timedelta(seconds=closest.covered - target_length)}"
        else:
            how = f"fell {timedelta(seconds=target_length - closest.covered)} short"
        lines.append(
            f"  The closest route covered {covered} of the {end - start} needed -- it {how} -- "
            f"reaching {closest.current_time}:"
        )
        for interval in closest.path[:8]:
            lines.append(f"    {interval}")
        if len(closest.path) > 8:
            lines.append(f"    ... and {len(closest.path) - 8} more")
        lines.append(
            "  The window has to be a union of whole archived intervals; check that its "
            "length and end time line up with the steps the source data actually holds."
        )

    if gave_up is not None:
        visited, reached = gave_up
        lines.append(
            f"  The search gave up after {visited} states, having reached {reached}: the "
            "description keeps offering intervals, but none of them closes the window."
        )
    return "\n".join(lines)
