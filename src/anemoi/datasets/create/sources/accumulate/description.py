# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Factorised archive descriptions for the accumulate source.

This module implements the recipe-facing *description keys* of the
accumulate source:

- ``from-trajectories:`` — the archive contains forecast trajectories,
  described by ``base_dates`` (recurring initialisation times), ``steps``
  (the lead-time grid) and ``accumulated`` (the archive's native
  accumulation scheme).  :class:`TrajectoryIntervalGenerator` turns such a
  description into candidate intervals for the covering search.
- ``from-increments:`` — base-less, valid-time-indexed archives storing
  fixed-length increments (handled in ``interval_generators.py``).
- ``from-lookup-table:`` — the explicit cycle lookup table
  (``CycleIntervalProvider`` in ``interval_generators.py``).

It also hosts :class:`AccumulateSchema`, the pydantic model attached to
``AccumulateSource.schema`` so recipes are validated at recipe time, and
the table of factorised descriptions for well-known MARS archives used by
``from-trajectories: auto``.
"""

from __future__ import annotations

import datetime
import logging
import re
import warnings
from typing import Any
from typing import Literal

import numpy as np
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.time_schemas import Frequency
from anemoi.datasets.create.time_schemas import Steps

from .covering_intervals import covering_intervals
from .interval_generators import IntervalGenerator

LOG = logging.getLogger(__name__)

MIGRATE_HINT = "run 'anemoi-datasets recipe --migrate <recipe>' to rewrite the recipe"

# ---------------------------------------------------------------------------
# The `accumulated:` scheme grammar
# ---------------------------------------------------------------------------

ACCUMULATED_VALUES = "'from-zero', 'from-previous-step', 'from-zero-reset-every-<frequency>' (e.g. 'from-zero-reset-every-24h')"

_RESET_RE = re.compile(r"^from-zero-reset-every-(.+)$")


def parse_accumulated(value: str) -> tuple[str, int | None]:
    """Parse an ``accumulated:`` scheme value.

    Parameters
    ----------
    value
        One of ``from-zero``, ``from-previous-step`` or
        ``from-zero-reset-every-<frequency>``.

    Returns
    -------
    tuple
        ``(kind, reset_hours)`` where *kind* is ``"from-zero"``,
        ``"from-previous-step"`` or ``"from-zero-reset"`` and
        *reset_hours* is the reset frequency in whole hours (``None``
        unless *kind* is ``"from-zero-reset"``).
    """
    if not isinstance(value, str):
        raise ValueError(f"Invalid 'accumulated' value {value!r}; expected one of {ACCUMULATED_VALUES}")
    if value == "from-zero":
        return "from-zero", None
    if value == "from-previous-step":
        return "from-previous-step", None
    m = _RESET_RE.match(value)
    if m:
        try:
            reset = frequency_to_timedelta(m.group(1))
        except Exception as e:
            raise ValueError(f"Invalid reset frequency in 'accumulated: {value}': {m.group(1)!r}") from e
        hours = reset.total_seconds() / 3600
        if not (hours.is_integer() and hours > 0):
            raise ValueError(f"Reset frequency in 'accumulated: {value}' must be a positive whole number of hours")
        return "from-zero-reset", int(hours)
    raise ValueError(f"Invalid 'accumulated' value {value!r}; expected one of {ACCUMULATED_VALUES}")


# ---------------------------------------------------------------------------
# base_dates: the recurring form (+ wildcard-string sugar)
# ---------------------------------------------------------------------------

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _coerce_time(value: Any) -> datetime.time:
    """Coerce ``6``, ``600`` (HHMM), ``"6"``, ``"06:00"`` or a ``datetime.time`` to a time."""
    if isinstance(value, datetime.time):
        return value
    if isinstance(value, int):
        if 0 <= value < 24:
            return datetime.time(value)
        if 100 <= value <= 2359 and value % 100 < 60:
            # MARS HHMM convention (e.g. 600, 1800)
            return datetime.time(value // 100, value % 100)
        raise ValueError(
            f"Invalid time value {value!r} (expected e.g. 6, 600, '06:00'). "
            "Note that YAML parses an unquoted HH:MM as a number — quote it: '18:00'"
        )
    if isinstance(value, str):
        s = value.strip()
        parts = s.split(":")
        if len(parts) == 1:
            return _coerce_time(int(parts[0]))
        return datetime.time(int(parts[0]), int(parts[1]))
    raise ValueError(f"Invalid time value {value!r} (expected e.g. 6, '06:00')")


def _wildcards_to_selectors(patterns: list[str]) -> dict:
    """Convert fnmatch-style wildcard pattern(s) to the structured selectors.

    The dialect is the one used by the ``from-trajectories`` source:
    patterns match ``"%Y-%m-%d %H:%M"`` formatted basetimes.  Only what the
    structured form can express is accepted (wildcard year/month, wildcard
    or concrete day, concrete time); anything else must be written in the
    structured form directly.
    """
    times: list[datetime.time] = []
    days: set[int] = set()
    day_wild: bool = False

    for pattern in patterns:
        p = pattern.strip()
        if " " not in p:
            raise ValueError(
                f"Invalid base_dates pattern {pattern!r}: expected '<date> <time>' " "(e.g. '????-??-?? 06:00')"
            )
        date_part, time_part = p.split(" ", 1)
        time_part = time_part.strip()
        if "?" in time_part or "*" in time_part:
            raise ValueError(
                f"Invalid base_dates pattern {pattern!r}: the time part must be concrete "
                "(one pattern per initialisation time); use the structured form otherwise"
            )
        tokens = date_part.split("-")
        if len(tokens) != 3:
            raise ValueError(f"Invalid base_dates pattern {pattern!r}: date part must be 'YYYY-MM-DD' shaped")
        year, month, day = tokens
        if year != "????" or month != "??":
            raise ValueError(
                f"Invalid base_dates pattern {pattern!r}: year and month must be wildcards "
                "('????-??-…'); use 'start'/'end' in the structured form to bound the archive"
            )
        if day == "??":
            day_wild = True
        elif "?" in day or "*" in day:
            raise ValueError(
                f"Invalid base_dates pattern {pattern!r}: the day must be '??' or concrete; "
                "use the structured form ('day_of_month:') otherwise"
            )
        else:
            days.add(int(day))
        t = time_part.split(":")
        times.append(datetime.time(int(t[0]), int(t[1]) if len(t) > 1 else 0))

    if days and day_wild:
        raise ValueError(
            "Inconsistent base_dates patterns: mixing wildcard days ('??') with "
            "concrete days; use the structured form instead"
        )

    result: dict[str, Any] = {"times": times}
    if days:
        result["day_of_month"] = sorted(days)
    return result


class RecurringBaseDates(BaseModel):
    """Recurring base-date (forecast initialisation time) selectors.

    Describes *which base dates exist* in an archive: the initialisation
    ``times`` (required), optionally restricted to given days of the month
    or week, optionally bounded by ``start``/``end`` for archives that only
    cover a fixed span.  A wildcard string (fnmatch dialect on
    ``"%Y-%m-%d %H:%M"``, as in the ``from-trajectories`` source) is
    accepted as sugar and coerced to the structured form.
    """

    model_config = ConfigDict(extra="forbid")

    times: list[datetime.time]
    day_of_month: list[int] | None = None
    day_of_week: list[str] | None = None
    start: datetime.datetime | None = None
    end: datetime.datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_wildcards(cls, data: Any) -> Any:
        if isinstance(data, str):
            data = [data]
        if isinstance(data, list):
            if not all(isinstance(x, str) for x in data):
                raise ValueError(f"Invalid base_dates {data!r}: expected a mapping or wildcard pattern string(s)")
            return _wildcards_to_selectors(data)
        return data

    @field_validator("times", mode="before")
    @classmethod
    def _coerce_times(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            value = [value]
        times = [_coerce_time(v) for v in value]
        if not times:
            raise ValueError("base_dates.times must not be empty")
        if len(set(times)) != len(times):
            raise ValueError(f"base_dates.times contains duplicates: {value}")
        return sorted(times)

    @field_validator("day_of_month", mode="before")
    @classmethod
    def _coerce_day_of_month(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            value = [value]
        days = [int(v) for v in value]
        for d in days:
            if not 1 <= d <= 31:
                raise ValueError(f"base_dates.day_of_month must be in 1..31, got {d}")
        return sorted(set(days))

    @field_validator("day_of_week", mode="before")
    @classmethod
    def _coerce_day_of_week(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            value = [value]
        result = []
        for v in value:
            name = str(v).strip().lower()[:3]
            if name not in WEEKDAYS:
                raise ValueError(f"base_dates.day_of_week entries must be one of {WEEKDAYS}, got {v!r}")
            result.append(name)
        return sorted(set(result), key=WEEKDAYS.index)

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce_bounds(cls, value: Any) -> Any:
        if value is None:
            return None
        return as_datetime(value)

    def matches(self, base: datetime.datetime) -> bool:
        """Return True if *base* is one of the base dates this selector describes."""
        if base.time() not in self.times:
            return False
        if self.day_of_month is not None and base.day not in self.day_of_month:
            return False
        if self.day_of_week is not None and WEEKDAYS[base.weekday()] not in self.day_of_week:
            return False
        if self.start is not None and base < self.start:
            return False
        if self.end is not None and base > self.end:
            return False
        return True


# ---------------------------------------------------------------------------
# from-trajectories: the archive description
# ---------------------------------------------------------------------------


class FromTrajectories(BaseModel):
    """The ``from-trajectories:`` archive description.

    The archive contains forecast trajectories initialised at
    ``base_dates``, with the ``steps`` lead-time grid, whose accumulated
    fields follow the ``accumulated`` scheme.  ``steps`` reuses the
    trajectory-layout :class:`~anemoi.datasets.create.recipe.dates.Steps`
    schema; a list of ranges is accepted for irregular grids.
    """

    model_config = ConfigDict(extra="forbid")

    base_dates: RecurringBaseDates
    steps: list[Steps]
    accumulated: str

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, value: Any) -> Any:
        if isinstance(value, dict) or isinstance(value, Steps):
            value = [value]
        return value

    @field_validator("accumulated")
    @classmethod
    def _check_accumulated(cls, value: str) -> str:
        parse_accumulated(value)
        return value

    @model_validator(mode="after")
    def _check_grid(self) -> "FromTrajectories":
        grid = self.step_grid_hours
        kind, _ = parse_accumulated(self.accumulated)
        if kind == "from-previous-step" and len(grid) < 2:
            raise ValueError("from-trajectories: 'accumulated: from-previous-step' needs at least two steps")
        if kind in ("from-zero", "from-zero-reset") and not any(s > 0 for s in grid):
            raise ValueError(f"from-trajectories: 'accumulated: {self.accumulated}' needs at least one step > 0")
        return self

    @property
    def step_grid_hours(self) -> list[int]:
        """The sorted union of the step ranges, in whole hours."""
        grid: set[int] = set()
        for r in self.steps:
            for value in r.values:
                grid.add(int(value / np.timedelta64(1, "h")))
        return sorted(grid)

    def step_pairs(self) -> list[tuple[int, int]]:
        """The archived ``(start_step, end_step)`` pairs implied by the description."""
        kind, reset = parse_accumulated(self.accumulated)
        grid = self.step_grid_hours
        if kind == "from-previous-step":
            return [(a, b) for a, b in zip(grid[:-1], grid[1:])]
        if kind == "from-zero":
            return [(0, s) for s in grid if s > 0]
        # from-zero-reset: each step is accumulated since the last reset of lead time
        return [((s - 1) // reset * reset, s) for s in grid if s > 0]


class TrajectoryIntervalGenerator(IntervalGenerator):
    """Candidate-interval generator for a ``from-trajectories:`` description.

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


# ---------------------------------------------------------------------------
# Factorised descriptions of well-known MARS archives (from-trajectories: auto)
# ---------------------------------------------------------------------------


def _mars_archive_description(_class: str, _stream: str | None = None, _origin: str | None = None) -> dict:
    """Return the factorised archive description for a well-known MARS archive.

    Parameters
    ----------
    _class
        MARS class (e.g., 'ea', 'od', 'rr', 'l5').
    _stream
        MARS stream (e.g., 'oper', 'enda', 'elda', 'enfo'). Defaults to 'oper'.
    _origin
        MARS origin (e.g., 'se-al-ec', 'fr-ms-ec'). Defaults to None.

    Returns
    -------
    dict
        A ``from-trajectories:`` payload.

    Raises
    ------
    NotImplementedError
        If the combination is not yet implemented.
    ValueError
        If the combination is unknown.
    """
    _stream = _stream or "oper"

    match (_class, _stream, _origin):
        case ("ea", "oper", _) | ("e6", "oper", _) | ("e6", "enda", _):
            return {
                "base_dates": {"times": [6, 18]},
                "steps": {"start": "0h", "end": "18h", "frequency": "1h"},
                "accumulated": "from-previous-step",
            }
        case ("ea", "enda", _):
            return {
                "base_dates": {"times": [6, 18]},
                "steps": {"start": "0h", "end": "18h", "frequency": "3h"},
                "accumulated": "from-previous-step",
            }
        case ("od", "oper", _):
            # https://apps.ecmwf.int/mars-catalogue/?stream=oper&levtype=sfc&time=00%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-25&type=fc&class=od
            return {
                "base_dates": {"times": [0, 12]},
                "steps": {"start": "1h", "end": "90h", "frequency": "1h"},
                "accumulated": "from-zero",
            }
        case ("od", "elda", _):
            # https://apps.ecmwf.int/mars-catalogue/?stream=elda&levtype=sfc&time=06%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-31&type=fc&class=od
            return {
                "base_dates": {"times": [6, 18]},
                "steps": {"start": "1h", "end": "12h", "frequency": "1h"},
                "accumulated": "from-zero",
            }
        case ("od", "enfo", _):
            # https://apps.ecmwf.int/mars-catalogue/?class=od&stream=enfo&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-31&time=06:00:00
            raise NotImplementedError("od-enfo archive description not implemented yet")

        case ("rr", _, "se-al-ec"):
            # https://apps.ecmwf.int/mars-catalogue/?class=rr&expver=prod&origin=se-al-ec&stream=oper&type=fc&year=2020&month=aug&levtype=sfc
            return {
                "base_dates": {"times": [0]},
                "steps": [
                    {"start": "1h", "end": "6h", "frequency": "1h"},
                    {"start": "6h", "end": "30h", "frequency": "3h"},
                ],
                "accumulated": "from-zero",
            }
        case ("rr", _, "fr-ms-ec"):
            # https://apps.ecmwf.int/mars-catalogue/?origin=fr-ms-ec&stream=oper&levtype=sfc&time=06%3A00%3A00&expver=prod&month=aug&year=2020&date=2020-08-31&type=fc&class=rr
            return {
                "base_dates": {"times": [0]},
                "steps": {"start": "1h", "end": "19h", "frequency": "3h"},
                "accumulated": "from-zero",
            }

        case ("l5", "oper", _):
            # https://apps.ecmwf.int/mars-catalogue/?class=l5&stream=oper&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-25&time=00:00:00
            return {
                "base_dates": {"times": [0]},
                "steps": {"start": "0h", "end": "24h", "frequency": "1h"},
                "accumulated": "from-previous-step",
            }

        case _:
            raise ValueError(f"Unknown MARS configuration: class={_class}, stream={_stream}, origin={_origin}")


def infer_from_trajectories(source_name: str | None, source: dict | None) -> FromTrajectories:
    """Infer a ``from-trajectories:`` description from a MARS source config (the ``auto`` sugar)."""
    assert None not in (source_name, source), "Source must be specified when using 'from-trajectories: auto'"
    if source_name != "mars":
        raise ValueError(
            "'from-trajectories: auto' is only supported for the 'mars' source; "
            "write the description explicitly for other sources"
        )

    _class, _stream, _origin = source.get("class"), source.get("stream"), source.get("origin")

    if _class is None:
        raise ValueError(
            "'from-trajectories: auto' infers the archive description from the mars "
            "source, but the mars source has no 'class'"
        )

    if (_stream is None) or (_origin is None):
        LOG.warning(
            f"Stream and/or origin unspecified for class {_class}, " f"stream and/or origin will be set as defaults.",
        )

    return FromTrajectories.model_validate(_mars_archive_description(_class, _stream, _origin))


# ---------------------------------------------------------------------------
# AccumulateSchema — recipe-time validation of the accumulate block
# ---------------------------------------------------------------------------

DESCRIPTION_KEYS = ("from-trajectories", "from-increments", "from-lookup-table")

# Sources that anchor fields to a model run: their intervals carry a basetime,
# which contradicts the base-less `from-increments:` description.
_BASE_ANCHORED_SOURCES = ("mars", "fdb")


def _hyphen_alias(name: str) -> str:
    return name.replace("_", "-")


class AccumulateSchema(BaseModel):
    """Validation schema for the ``accumulate`` source in recipes.

    Archive recipes carry **exactly one** description key
    (``from-trajectories:`` / ``from-increments:`` / ``from-lookup-table:``);
    trajectory-layout recipes carry none and declare ``accumulated:`` at
    block level instead (this cross-layout rule is enforced by the
    ``Recipe`` model, which knows the output layout).  The pre-redesign
    spellings (``covering:``, ``availability:``, ``accumulation:``) are
    accepted for one release with a ``DeprecationWarning``.
    """

    model_config = ConfigDict(
        alias_generator=_hyphen_alias,
        populate_by_name=True,
        extra="forbid",
    )

    period: Frequency
    source: dict[str, Any]

    from_trajectories: FromTrajectories | Literal["auto"] | None = None
    from_increments: Frequency | None = None
    from_lookup_table: dict[str, Any] | None = None

    accumulated: str | None = None
    patch: list[str] | None = None
    group_by: dict[str, Any] | None = None

    # Deprecated spellings, kept for one release. `accumulation` and
    # `availability` are folded into their replacements during validation
    # (so they are always None afterwards) and excluded from dumps.
    accumulation: str | None = Field(default=None, exclude=True)
    covering: Any = None
    availability: Any = Field(default=None, exclude=True)

    @property
    def description_key(self) -> str | None:
        """The hyphenated name of the description key in use, or None."""
        if self.from_trajectories is not None:
            return "from-trajectories"
        if self.from_increments is not None:
            return "from-increments"
        if self.from_lookup_table is not None:
            return "from-lookup-table"
        return None

    @model_validator(mode="after")
    def _check(self) -> "AccumulateSchema":
        if not (isinstance(self.source, dict) and len(self.source) == 1):
            raise ValueError(f"accumulate: 'source' must have exactly one key, got {list(self.source.keys())}")

        # -- deprecated aliases -------------------------------------------
        if self.accumulation is not None:
            if self.accumulated is not None:
                raise ValueError(
                    "accumulate: cannot specify both 'accumulated' and its deprecated alias 'accumulation'"
                )
            warnings.warn(
                "'accumulation:' is deprecated; use 'accumulated:' instead " f"({MIGRATE_HINT}).",
                DeprecationWarning,
                stacklevel=2,
            )
            self.accumulated = self.accumulation
            self.accumulation = None

        if self.availability is not None:
            if self.covering is not None:
                raise ValueError(
                    "accumulate: cannot specify both 'covering' and its deprecated alias 'availability'"
                )
            self.covering = {"auto": self.availability}
            self.availability = None

        if self.covering is not None:
            warnings.warn(
                "'covering:'/'availability:' are deprecated; describe the archive with "
                "'from-trajectories:', 'from-increments:' or 'from-lookup-table:' instead "
                f"({MIGRATE_HINT}).",
                DeprecationWarning,
                stacklevel=2,
            )

        # -- exactly one description --------------------------------------
        given = [k for k in DESCRIPTION_KEYS if getattr(self, k.replace("-", "_")) is not None]
        if self.covering is not None:
            given = given + ["covering"]
        if len(given) > 1:
            raise ValueError(f"accumulate: only one archive description is allowed, got {given}")

        if self.accumulated is not None:
            if given:
                raise ValueError(
                    f"accumulate: '{given[0]}' and block-level 'accumulated:' are mutually "
                    "exclusive — in archive recipes 'accumulated:' belongs inside "
                    "'from-trajectories:'; bare 'accumulated:' is for trajectory-layout recipes"
                )
            parse_accumulated(self.accumulated)

        if not given and self.accumulated is None:
            raise ValueError(
                "accumulate: describe the archive with exactly one of "
                f"{', '.join(repr(k) for k in DESCRIPTION_KEYS)} — or, in a "
                "'layout: trajectories' recipe, declare 'accumulated:' "
                f"(one of {ACCUMULATED_VALUES})"
            )

        # -- per-description rules ----------------------------------------
        if self.from_increments is not None:
            hours = self.from_increments.total_seconds() / 3600
            if not (hours.is_integer() and hours > 0):
                raise ValueError("accumulate: 'from-increments' must be a positive whole number of hours")
            if self.period % self.from_increments != datetime.timedelta(0):
                raise ValueError(
                    f"accumulate: 'from-increments' ({frequency_to_string(self.from_increments)}) "
                    f"must divide 'period' ({frequency_to_string(self.period)})"
                )
            if datetime.timedelta(hours=24) % self.from_increments != datetime.timedelta(0):
                raise ValueError(
                    f"accumulate: 'from-increments' ({frequency_to_string(self.from_increments)}) must divide 24h"
                )
            source_name = next(iter(self.source))
            if source_name in _BASE_ANCHORED_SOURCES:
                raise ValueError(
                    f"accumulate: 'from-increments' describes a base-less, valid-time-indexed "
                    f"archive, but '{source_name}' fields are anchored to a model run — "
                    "use 'from-trajectories:' instead"
                )

        if self.patch is not None:
            from .field_to_interval import patch_registry

            for key in self.patch:
                if key not in patch_registry:
                    raise ValueError(
                        f"accumulate: unknown patch {key!r} (expected one of {sorted(patch_registry)})"
                    )

        return self
