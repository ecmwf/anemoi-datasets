# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``base_dates``: the recurring form (+ wildcard-string sugar)."""

from __future__ import annotations

import datetime
from typing import Any

from anemoi.utils.dates import as_datetime
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEKDAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


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
    pairs: set[tuple[int | None, datetime.time]] = set()
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
                "('????-??-…'); use 'start'/'end' in the structured form to bound them"
            )
        if day == "??":
            day_wild = True
            day_value = None
        elif "?" in day or "*" in day:
            raise ValueError(
                f"Invalid base_dates pattern {pattern!r}: the day must be '??' or concrete; "
                "use the structured form ('day_of_month:') otherwise"
            )
        else:
            day_value = int(day)
        t = time_part.split(":")
        pairs.add((day_value, datetime.time(int(t[0]), int(t[1]) if len(t) > 1 else 0)))

    days = sorted({d for d, _ in pairs if d is not None})
    times = sorted({t for _, t in pairs})

    if days and day_wild:
        raise ValueError(
            "Inconsistent base_dates patterns: mixing wildcard days ('??') with "
            "concrete days; use the structured form instead"
        )

    # The structured form means the cross product day_of_month × times; a
    # pattern set like ['????-??-01 06:00', '????-??-15 18:00'] does not
    # factorise into it and must not be silently reinterpreted.
    if days and {(d, t) for d in days for t in times} != pairs:
        raise ValueError(
            f"base_dates patterns {patterns!r} do not factorise into "
            "day_of_month × times; use the structured form instead"
        )

    result: dict[str, Any] = {"times": times}
    if days:
        result["day_of_month"] = days
    return result


class RecurringBaseDates(BaseModel):
    """Recurring base-date (forecast initialisation time) selectors.

    Describes *which base dates exist* in the source data: the initialisation
    ``times`` (required), optionally restricted to given days of the month
    or week, optionally bounded by ``start``/``end`` for data that only
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
            # exact 3-letter abbreviation or exact full name — no prefix
            # matching, so a misspelling like 'mondi' is rejected
            name = str(v).strip().lower()
            if name in WEEKDAYS:
                result.append(name)
            elif name in WEEKDAY_NAMES:
                result.append(name[:3])
            else:
                raise ValueError(
                    f"base_dates.day_of_week entries must be one of {WEEKDAYS} or the full names, got {v!r}"
                )
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
