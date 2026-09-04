# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re
from datetime import datetime
from datetime import timedelta


def step_to_timedelta(step: str | int | timedelta) -> timedelta:
    """Parse a step into a timedelta.

    Inverse of `timedelta_to_step`. A bare number means hours, so hour-based
    configs (``frequency: 1``, ``last_step: 24``) keep working

        12 -> 12h, "12" -> 12h, "24h" -> 24h
        "10m" -> 10min, "10h10m" -> 10h10min

    Raises
    ------
    ValueError
        If the step cannot be parsed.
    """
    if isinstance(step, timedelta):
        return step
    if isinstance(step, int):
        return timedelta(hours=step)

    step = str(step).strip()
    if step.isdigit():
        return timedelta(hours=int(step))

    match = re.match(r"^(?:(\d+)h)?(?:(\d+)m)?$", step)
    if not match or not any(match.groups()):
        raise ValueError(f"Cannot parse step {step!r}; expected forms like '12', '24h', '10m', '10h10m'.")
    hours, minutes = match.groups()
    return timedelta(hours=int(hours or 0), minutes=int(minutes or 0))


def timedelta_to_step(offset: timedelta) -> int | str:
    """Format a lead time as a step.

    Whole hours stay plain integers, so that requests built from hour-based
    recipes are byte-for-byte what they have always been. Only sub-hourly
    offsets need the string form, carrying a minute suffix:

        0:00  -> 0        12:00 -> 12
        0:10  -> "10m"    10:10 -> "10h10m"

    Raises
    ------
    ValueError
        If the offset is negative or not a whole number of minutes.
    """
    seconds = offset.total_seconds()
    if seconds < 0:
        raise ValueError(f"Step must not be negative, got {offset}.")
    if seconds % 60:
        raise ValueError(f"Step must be a whole number of minutes, got {offset}.")
    hours, minutes = divmod(int(seconds // 60), 60)
    if minutes == 0:
        return hours
    if hours == 0:
        return f"{minutes}m"
    return f"{hours}h{minutes}m"


class SignedInterval:
    def __init__(self, start: datetime, end: datetime, base: datetime | None = None):
        self.start = start
        self.end = end
        self.base = base

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

    # ------------------------------------------------------------------
    # Conceptual accessors used by the dispatch consumers.
    # ``base_time`` is the model-run time the interval is anchored to
    # (``None`` only for grib_index, which is base-less by construction).
    # ``valid_time`` is the validity time of the underlying archived
    # field, i.e. the later end of the interval regardless of its sign.
    # ------------------------------------------------------------------

    @property
    def valid_time(self) -> datetime:
        """Validity time of the underlying archived field (``max(start, end)``)."""
        return self.max

    def __neg__(self):
        return SignedInterval(start=self.end, end=self.start, base=self.base)

    def __eq__(self, other):
        if not isinstance(other, SignedInterval):
            return NotImplemented
        if self.start != other.start or self.end != other.end:
            return False
        if self.base != other.base:
            return False
        return True

    def __hash__(self):
        return hash((self.start, self.end, self.base))

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
                    timedelta_to_step(self.start - self.base),
                    timedelta_to_step(self.end - self.base),
                ]
            else:
                earlier = self.end - self.base
                steps = [
                    timedelta_to_step(earlier) if not earlier else f"-{timedelta_to_step(earlier)}",
                    timedelta_to_step(self.start - self.base),
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
