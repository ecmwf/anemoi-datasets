# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""How a window is resolved into parts, and how arriving fields are matched to them.

Everything else about a window source is shared -- fetching, grouping, reducing,
completeness, output -- so this is the whole of the difference between the two
families, and :class:`~.source.WindowSourceBase` drives its loop through it.

Two implementations, discriminated by what ``from:`` says the source data is:

- ``from: {accumulation: ...}`` -- fields that each **span** an interval, so a window
  is covered by archived intervals and a field is identified by the interval it
  spans.  Used by ``accumulate``.
- ``from: {frequency: ...}`` -- fields that **exist every** ``frequency``, so a window
  is a list of sample times and a field is identified by its validity time.  Used by
  ``average`` / ``minimum`` / ``maximum``.

A *part* is one of those: an archived interval in the first case, a sample
time in the second.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from .samples import Sample
from .subwindows import Subwindow

#: A build target: ``(valid_date, basetime)``, with *basetime* ``None`` outside a
#: trajectory layout.
Target = tuple

#: One of the things a window is made of. The two are not alike: subwindows *partition*
#: the window, samples are instants inside it and partition nothing. What they share is
#: that each yields one value and one weight, which is all a reduction needs.
Part = Subwindow | Sample


class WindowPlan(ABC):
    """Resolve windows into parts, and match arriving fields to them."""

    @abstractmethod
    def parts_for(self, targets: list[Target]) -> dict[Target, list[Part]]:
        """The parts each target's window is made of, keyed by target."""

    @abstractmethod
    def argument(self, targets: list[Target], parts: dict[Target, list[Part]]) -> Any:
        """What to ask the subsource for (a ``ValidDates``/``Intervals``/... argument)."""

    @abstractmethod
    def identify(self, field: Any) -> Any:
        """What this field *is*, in whatever terms the parts are expressed."""

    @abstractmethod
    def new_reducer(self, target: Target, key: tuple, parts: list[Part]) -> Any:
        """A reducer for one ``(target, variable)``, expecting *parts*."""

    @abstractmethod
    def offer(self, reducer: Any, values: Any, identity: Any) -> bool:
        """Offer a field to a reducer; True if it was needed."""

    def candidates(self, identity: Any, targets: list[Target]) -> list[Target]:
        """The targets whose window might contain *identity*.

        The default is "all of them", leaving the reducer to decide -- which is what
        interval matching needs, because an interval is matched on its endpoints and
        base rather than looked up. A plan whose identities are hashable should build
        a reverse index in :meth:`parts_for` and narrow this down.
        """
        return targets

    # ── optional diagnostics ─────────────────────────────────────────
    #
    # A window source that cannot place a field has almost nothing to say about why,
    # so `accumulate` keeps a running log of what arrived and where it went. These
    # hooks let a plan collect that without the loop knowing about it.

    def begin(self, input_fields: Any, reducers: dict) -> None:
        """Called once, before the first field."""

    def note(self, field: Any, identity: Any, used_by: list) -> None:
        """Called once per field, with the ``(target, reducer)`` pairs that took it."""

    def unused_field(self, field: Any, identity: Any) -> None:
        """Called when no window wanted a field. Must raise."""
        raise ValueError(f"Field {field} ({identity}) is not needed by any window.")
