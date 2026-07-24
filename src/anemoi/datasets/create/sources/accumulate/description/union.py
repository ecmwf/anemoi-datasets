# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``from:`` — the source-data description, recognised structurally.

There is no ``type:`` key: a ``from:`` block is a :class:`FromLookupTable`
when it carries ``lookup-table``, a :class:`FromTrajectories` when it
carries ``base_dates``/``steps``, and a :class:`FromBare` otherwise.
"""

from __future__ import annotations

import datetime
from typing import Any
from typing import Literal

import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import TypeAdapter
from pydantic import field_validator
from pydantic import model_validator

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.time_schemas import Steps

from ..covering_intervals import covering_intervals
from ..interval_generators import IntervalGenerator
from .accumulation import parse_accumulation
from .base_dates import RecurringBaseDates


def _parse_step_pair(item: Any) -> tuple[int, int]:
    """Parse one explicit step pair into whole-hour ``(start, end)``.

    Two spellings are accepted: the string ``"sA-sE"`` (e.g. ``"6-9"``) and a
    two-element list/tuple ``[sA, sE]`` (e.g. ``[6, 9]``).
    """
    if isinstance(item, str):
        if item.count("-") != 1:
            raise ValueError(f"from: step pair {item!r} must be 'startStep-endStep' (whole hours), e.g. '6-9'")
        a, b = item.split("-")
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        a, b = item
    else:
        raise ValueError(f"from: step pair {item!r} must be 'sA-sE' or [sA, sE] (whole hours), e.g. '6-9' or [6, 9]")
    try:
        start, end = int(a), int(b)
    except (ValueError, TypeError):
        raise ValueError(f"from: step pair {item!r} must be whole hours, e.g. '6-9' or [6, 9]")
    if not (0 <= start < end):
        raise ValueError(f"from: step pair {item!r} must have 0 <= startStep < endStep")
    return start, end


#: Sentinel on ``base_dates`` and ``steps`` meaning "inherit the run grid from
#: the output layout" (only meaningful under a ``layout: trajectories`` recipe).
FROM_LAYOUT = "from-layout"


class FromTrajectories(BaseModel):
    """``from:`` with ``base_dates`` + ``steps`` — source data made of forecast runs.

    The source data contains forecast trajectories initialised at
    ``base_dates``, described by ``steps``.  ``steps`` is written one of three
    ways:

    - a **regular range** (``{start, end, frequency}``, the trajectory-layout
      :class:`~anemoi.datasets.create.time_schemas.Steps` schema) *plus* an
      ``accumulation`` scheme (``from-zero`` / a duration /
      ``from-zero-reset-every-<freq>``) that says how each field accumulates;
    - an **explicit list of pairs**, one per available field, each written
      ``"sA-sE"`` or ``[sA, sE]`` — the pairs *are* the description, so
      ``accumulation`` is not used (and forbidden). This is the only form
      general enough for an irregular grid of mixed accumulation lengths
      (e.g. 1 h increments to 6 h, 3 h after);
    - the sentinel ``from-layout`` (on **both** ``base_dates`` and ``steps``)
      *plus* an ``accumulation`` scheme — the subsource *is* the run the
      output layout imposes, so the grid comes from the layout at runtime and
      only the accumulation scheme is stated here.  Valid only under a
      ``layout: trajectories`` recipe (:attr:`is_layout_grid`).

    ``from:`` describes the subsource; the output layout decides the output.
    They are orthogonal — the two are bridged at runtime — so a
    :class:`FromTrajectories` block is accepted in *both* output layouts.
    """

    model_config = ConfigDict(extra="forbid")

    accumulation: str | None = None
    base_dates: Literal["from-layout"] | RecurringBaseDates
    steps: Literal["from-layout"] | Steps | list[Any]

    @property
    def is_layout_grid(self) -> bool:
        """True when the run grid is inherited from the output layout (``from-layout``)."""
        return self.base_dates == FROM_LAYOUT

    @property
    def is_explicit_pairs(self) -> bool:
        """True when ``steps`` is an explicit ``"sA-sE"`` pair list rather than a range."""
        return isinstance(self.steps, list)

    @field_validator("accumulation")
    @classmethod
    def _check_accumulation(cls, value: str | None) -> str | None:
        if value is not None:
            parse_accumulation(value)
        return value

    @model_validator(mode="after")
    def _check_grid(self) -> "FromTrajectories":
        # `from-layout` is a paired sentinel: it must sit on both `base_dates`
        # and `steps` together (the layout supplies the whole run grid).
        base_layout = self.base_dates == FROM_LAYOUT
        step_layout = self.steps == FROM_LAYOUT
        if base_layout != step_layout:
            raise ValueError(
                "from: 'from-layout' must be set on both 'base_dates' and 'steps' together — "
                "the output layout supplies the whole run grid, or neither key does"
            )
        if base_layout:
            # The layout imposes the grid; only the accumulation scheme is
            # needed (any of from-zero / a duration / from-zero-reset).
            if self.accumulation is None:
                raise ValueError(
                    "from: 'base_dates: from-layout, steps: from-layout' needs an 'accumulation' "
                    "(how the imposed run accumulates: 'from-zero', a duration, or "
                    "'from-zero-reset-every-<freq>')"
                )
            return self

        if self.is_explicit_pairs:
            # The pairs are the whole description; a scheme would contradict them.
            if self.accumulation is not None:
                raise ValueError(
                    "from: an explicit 'steps' pair list is the full description — "
                    "remove 'accumulation' (each 'sA-sE' pair already states its own window)"
                )
            pairs = [_parse_step_pair(s) for s in self.steps]
            if not pairs:
                raise ValueError("from: 'steps' is empty")
            if sorted(pairs, key=lambda p: (p[1], p[0])) != pairs:
                raise ValueError("from: explicit 'steps' pairs must be ordered (ascending by end step, then start)")
            if len(set(pairs)) != len(pairs):
                raise ValueError("from: explicit 'steps' pairs must not repeat")
            return self

        # Range form: a scheme is required to say how each field accumulates.
        if self.accumulation is None:
            raise ValueError(
                "from: a 'steps' range needs an 'accumulation' (how each field accumulates: "
                "'from-zero', a duration, or 'from-zero-reset-every-<freq>'); "
                "or give an explicit 'steps' pair list instead"
            )
        grid = self.step_grid_hours
        kind, hours = parse_accumulation(self.accumulation)

        if kind == "increment":
            # A duration is the window *length* each field holds — the field at
            # step *s* covers ``(s − accumulation, s)``. It is independent of
            # ``frequency`` (the step *spacing*): equal = contiguous increments,
            # accumulation > frequency = overlapping/rolling windows,
            # accumulation < frequency = a sparse grid with gaps. The only
            # constraint is that the first field cannot start before the
            # forecast. (For a mixed-*length* grid, use an explicit pair list.)
            first = int(self.steps.start.total_seconds() // 3600)
            if first < hours:
                raise ValueError(
                    f"from: 'steps.start' ({first}h) is shorter than the 'accumulation' length "
                    f"({hours}h); the first field would start before the forecast"
                )
        if not grid:
            raise ValueError("from: 'steps' is empty")
        if grid[0] == 0:
            # a per-step duration is rejected above with a more specific hint.
            raise ValueError(
                "from: 'steps' lists the steps at which fields exist, and no field "
                "exists at step 0 under any scheme — start 'steps' at the first "
                "archived step"
            )
        return self

    @property
    def step_grid_hours(self) -> list[int]:
        """The sorted end-steps, in whole hours (range form only)."""
        if self.is_layout_grid:
            raise ValueError(
                "from: 'step_grid_hours' is not defined for a 'from-layout' description — "
                "the run grid comes from the output layout at runtime"
            )
        if self.is_explicit_pairs:
            return sorted({_parse_step_pair(s)[1] for s in self.steps})
        grid: set[int] = set()
        for value in self.steps.values:
            grid.add(int(value / np.timedelta64(1, "h")))
        return sorted(grid)

    def step_pairs(self) -> list[tuple[int, int]]:
        """The ``(start_step, end_step)`` pairs of the fields the source data holds.

        An explicit pair list *is* those pairs; a range + ``accumulation``
        derives them (``steps`` lists the end-steps at which fields exist,
        ``accumulation`` says what each one spans).
        """
        if self.is_layout_grid:
            raise ValueError(
                "from: 'step_pairs' is not defined for a 'from-layout' description — "
                "the run grid comes from the output layout at runtime"
            )
        if self.is_explicit_pairs:
            return sorted((_parse_step_pair(s) for s in self.steps), key=lambda p: (p[1], p[0]))
        kind, hours = parse_accumulation(self.accumulation)
        grid = self.step_grid_hours
        if kind == "increment":
            # One regular range, enforced above: each field covers the frequency
            # (== the declared duration) ending at its own step.
            return [(s - hours, s) for s in grid]
        if kind == "from-zero":
            return [(0, s) for s in grid]
        # from-zero-reset: each step is accumulated since the last reset of lead time
        return [((s - 1) // hours * hours, s) for s in grid]


class FromBare(BaseModel):
    """``from:`` with only ``accumulation`` — the run grid comes from context.

    A bare ``from:`` states just how the source data accumulates.  Its
    meaning is fixed by *where* it is used, not by the block:

    - in a ``layout: trajectories`` recipe it describes the imposed run
      (``from-zero``, a per-step duration, or
      ``from-zero-reset-every-<freq>``);
    - in any other layout it describes base-less, validity-time-indexed
      source data, and ``accumulation`` must then be a fixed duration — the
      window each field holds.
    """

    model_config = ConfigDict(extra="forbid")

    accumulation: str

    @field_validator("accumulation")
    @classmethod
    def _check_accumulation(cls, value: str) -> str:
        parse_accumulation(value)
        return value

    @property
    def kind(self) -> str:
        """One of ``"from-zero"``, ``"increment"`` or ``"from-zero-reset"``."""
        return parse_accumulation(self.accumulation)[0]

    @property
    def duration(self) -> datetime.timedelta | None:
        """The accumulation length, when it is a fixed duration (else ``None``)."""
        kind, hours = parse_accumulation(self.accumulation)
        return datetime.timedelta(hours=hours) if kind == "increment" else None


class FromLookupTable(BaseModel):
    """``from: {lookup-table: {...}}`` — the explicit cycle table.

    The escape hatch for step layouts that do not factorise into
    ``base_dates × steps``.  Entries are keyed by the window's offset
    inside a repeating cycle anchored at ``start``; they are passed
    through to :class:`LookupTableIntervalGenerator` unchanged.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    lookup_table: dict[str, Any] = Field(alias="lookup-table")

    def entries(self) -> dict[str, Any]:
        """The payload to hand to ``LookupTableIntervalGenerator``."""
        return dict(self.lookup_table)


#: Recognised structurally (see the module docstring) — no discriminator.
From = FromTrajectories | FromBare | FromLookupTable

#: Built once — a per-call model would rebuild pydantic's validator every time.
_FROM_ADAPTER: TypeAdapter[From] = TypeAdapter(From)


class TrajectoryIntervalGenerator(IntervalGenerator):
    """Candidate-interval generator for a :class:`FromTrajectories` ``from:`` description.

    Replaces the raw step-pair ``SearchableIntervalGenerator`` for the
    factorised form: candidates are computed *exactly* for a given time
    (no ``search_range`` window) from base_dates × steps × accumulated,
    and the Dijkstra search in ``covering_intervals`` is layered on top
    unchanged.
    """

    def __init__(self, description: FromTrajectories) -> None:
        self.description = description
        self.pairs = description.step_pairs()

    def covering_intervals(self, start: datetime.datetime, end: datetime.datetime) -> list[SignedInterval]:
        """Return available SignedIntervals covering the period start->end."""
        return covering_intervals(start, end, self)

    def __call__(self, current_time: datetime.datetime) -> list[SignedInterval]:
        """Generate the candidate intervals starting (or, negated, ending) at *current_time*."""
        base_dates = self.description.base_dates
        intervals: list[SignedInterval] = []
        seen: set[SignedInterval] = set()

        for start_step, end_step in self.pairs:
            # Forward: an interval starting at current_time has base = current_time - start_step.
            base = current_time - datetime.timedelta(hours=start_step)
            if base_dates.matches(base):
                interval = SignedInterval(
                    start=current_time,
                    end=base + datetime.timedelta(hours=end_step),
                    base=base,
                )
                if interval not in seen:
                    seen.add(interval)
                    intervals.append(interval)

            # Backward: an interval ending at current_time has base = current_time - end_step;
            # its negation starts at current_time (used for the signed walk).
            base = current_time - datetime.timedelta(hours=end_step)
            if base_dates.matches(base):
                interval = -SignedInterval(
                    start=base + datetime.timedelta(hours=start_step),
                    end=current_time,
                    base=base,
                )
                if interval not in seen:
                    seen.add(interval)
                    intervals.append(interval)

        # quite important to sort by reversed base to prioritise most recent base in case of ties
        return sorted(intervals, key=lambda x: -(x.base or x.start).timestamp())
