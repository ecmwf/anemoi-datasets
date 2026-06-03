# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime


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
