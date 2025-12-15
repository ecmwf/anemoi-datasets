# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from heapq import heappop
from heapq import heappush
from typing import Callable
from typing import List
from typing import Optional

LOG = logging.getLogger(__name__)


class SignedInterval:
    def __init__(self, start: datetime, end: datetime, **extras):
        self.start = start
        self.end = end
        self.base = extras.pop("base", None)
        self.extras = extras

    @property
    def length(self) -> float:
        """Length in seconds (can be negative)."""
        return (self.end - self.start).total_seconds()

    @property
    def sign(self) -> int:
        return 1 if self.length >= 0 else -1

    @property
    def min(self):
        return min(self.start, self.end)

    @property
    def max(self):
        return max(self.start, self.end)

    def __neg__(self):
        return SignedInterval(start=self.end, end=self.start, base=self.base, **self.extras)

    def __eq__(self, other):
        if not isinstance(other, SignedInterval):
            return NotImplemented
        if self.start != other.start or self.end != other.end or self.base != other.base:
            return False
        for k in set(self.extras) | set(other.extras):
            LOG.warning(f"Comparing key: {k} in {self.__class__.__name__}")
            if k == "base":
                continue
            if k not in self.extras or k not in other.extras:
                return False
            if self.extras[k] != other.extras[k]:
                return False
        return True

    def __hash__(self):
        return hash((self.start, self.end, self.base, tuple(sorted(self.extras.items()))))

    def __rich__(self):
        return self.__repr__(colored=True)

    def __repr__(self, colored: bool = False):
        try:
            # use frequency_to_string only if available
            # as this class should not depends on anemoi.utils
            from anemoi.utils.dates import frequency_to_string
        except ImportError:

            def frequency_to_string(delta):
                return str(delta)

        start = self.start.strftime("%Y%m%d.%H%M")
        end = self.end.strftime("%Y%m%d.%H%M")
        if start[:9] == end[:9]:
            end = " " * 9 + end[9:]

        if self.base is not None:
            base = self.base.strftime("%Y%m%d.%H%M")
            if self.sign > 0:
                steps = [
                    int((self.start - self.base).total_seconds() / 3600),
                    int((self.end - self.base).total_seconds() / 3600),
                ]
            else:
                steps = [
                    -int((self.end - self.base).total_seconds() / 3600),
                    int((self.start - self.base).total_seconds() / 3600),
                ]
            base_str = f", base={base}, [{steps[0]}-{steps[1]}]"
        else:
            base_str = ""

        if self.start < self.end:
            period = f"+{frequency_to_string(self.end - self.start)}"
        elif self.start == self.end:
            period = "0s"
        else:
            period = f"-{frequency_to_string(self.start - self.end)}"
        period = period.ljust(4)

        if colored:
            # using rich colors
            start = f"[blue]{start}[/blue]"
            end = f"[blue]{end}[/blue]"
            if self.start < self.end:
                period = f"[green]{period}[/green]"
            elif self.start == self.end:
                period = f"[yellow]{period}[/yellow]"
            else:
                period = f"[red]{period}[/red]"

        return f"SignedInterval({start}{period}->{end}{base_str} )"


@dataclass(order=True)
class HeapState:
    total_cost: float
    covered: float
    counter: int
    current_time: datetime
    current_base: Optional[datetime]
    path: List[SignedInterval] = field(compare=False)


def covering_intervals(
    start: datetime,
    end: datetime,
    candidates: Callable | list | tuple,
    /,
    switch_penalty: int = 24 * 3600 * 7,
    max_delta: timedelta = timedelta(hours=24 * 2),
    error_on_fail: bool = True,
) -> List[SignedInterval] | None:
    """Find a path of intervals covering [start, end] with minimal base switches, then minimal total absolute length.
    Uses a Dijkstra-like algorithm to find the optimal path.

    Args:
        start: Start datetime of the target interval.

        end: End datetime of the target interval.

        candidates: A function(current: datetime, current_base: Optional[datetime]) -> Iterable[SignedInterval]
            that provides candidate intervals covering the current time.

        switch_penalty: Penalty (in seconds) for switching bases between intervals.

        max_delta: Maximum allowed deviation from start/end for search.

        error_on_fail: Whether to raise an error if coverage cannot be found.

    Returns:
        A list of SignedInterval objects covering [start, end], or None if no coverage found and error_on_fail is False.

    """
    target_length = (end - start).total_seconds()

    pq: List[HeapState] = []  # pq: priority queue
    counter = itertools.count()
    heappush(
        pq,
        HeapState(total_cost=0.0, covered=0.0, counter=next(counter), current_time=start, current_base=None, path=[]),
    )

    visited: dict[tuple[datetime, Optional[datetime], float], float] = {}

    while pq:
        state = heappop(pq)
        key = (state.current_time, state.current_base, state.covered)

        if key in visited and state.total_cost >= visited[key]:
            continue
        visited[key] = state.total_cost

        # Goal: cumulative coverage matches target
        if state.covered == target_length:
            return state.path

        if (len(visited) > 1000) and (state.current_time > end + max_delta or state.current_time < start - max_delta):
            msg = f"Exceeded search limits: visited={len(visited)}, current_time={state.current_time}, target=({start} → {end}), max_delta={max_delta}"
            if error_on_fail:
                raise ValueError(msg)
            LOG.warning(msg)
            return None

        for interval in candidates(state.current_time, current_base=state.current_base, start=start, end=end):
            if interval.end == state.current_time:
                interval = -interval

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

    msg = f"Cannot find coverage of {start} → {end}"
    if error_on_fail:
        raise ValueError(msg)
    LOG.warning(msg)
    return None
