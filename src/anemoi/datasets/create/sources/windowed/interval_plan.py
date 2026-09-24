# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The plan for interval-valued source data -- ``from: {accumulation: ...}``.

A window is covered by archived intervals, and a field is identified by the interval it
spans. Unlike sample times those identities are not looked up but *matched*, on their
endpoints and base, with a fallback for a base-less generator -- so every target is
offered every field and the reducer decides.

That is also why this plan carries :class:`~.reducer.Logs`: when a field fits no window
there is nothing to point at, so it keeps a running account of what arrived and where it
went.

Used by ``accumulate`` and by the reductions over interval-valued source data, which is
why it lives here rather than beside either of them.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals

from .plan import Target
from .plan import WindowPlan
from .operations import Operation
from .operations import operation_factory
from .reducer import Logs
from .reducer import Reducer
from .states import SubwindowState
from .states import field_statistic
from .subwindows import contributions_of
from .subwindows import validate_partition

LOG = logging.getLogger(__name__)


def _unique(intervals) -> list:
    """The intervals, in order, with repeats dropped.

    Keyed on the *field* rather than the signed interval: ``+a(0,7)`` and ``-a(0,7)``
    are one retrieval, and neighbouring subwindows of a cumulative archive share an
    endpoint, so both spellings routinely appear in one window set.
    """
    seen: dict = {}
    for interval in intervals:
        seen.setdefault((interval.min, interval.max, interval.base), interval)
    return list(seen.values())


class IntervalPlan(WindowPlan):
    """Reduce the archived intervals covering each window.

    Parameters
    ----------
    period : datetime.timedelta
        The window length.
    covering : Any
        Resolves a window into the intervals covering it.
    source : dict
        The subsource config, for the diagnostics.
    field_to_interval : Any
        Maps an arriving field to the interval it spans.
    basetime : bool
        Whether the reducer should stamp its output with the target's basetime
        (a trajectory row) rather than the start of the window.
    operation : str or Operation, optional
        What reduces the subwindow values. Defaults to a sum.
    over : datetime.timedelta, optional
        The subwindow length; defaults to *period*, i.e. the window is one part. A
        shorter value cuts the window into ``over``-long parts and covers each on its
        own, instead of letting the search cover the whole window as cheaply as it
        can. That is what turns a cumulative archive into per-``over`` values to
        reduce -- "the wettest hour in the window".
    forecast_items : list, optional
        ``(valid_time, basetime)`` rows, when the subsource is the run the layout
        imposes; makes the argument a ``ForecastIntervals``.
    accumulating : bool, optional
        Whether the archive stores *accumulations*, so a subwindow carries a total
        rather than a statistic of its own. True when ``from:`` states an
        ``accumulation`` scheme; false for an explicit list of step pairs, which is
        how an archive of stored maxima is described. Only used by
        :meth:`_check_reducible`.
    """

    def __init__(
        self,
        period: datetime.timedelta,
        covering: Any,
        source: dict,
        field_to_interval: Any,
        basetime: bool = False,
        operation: Operation | str | None = None,
        over: datetime.timedelta | None = None,
        forecast_items: list | None = None,
        accumulating: bool = False,
    ) -> None:
        self.period = period
        self.covering = covering
        self.source = source
        self.field_to_interval = field_to_interval
        self.basetime = basetime
        self.operation = operation_factory(operation)
        # Unstated, the subwindow length is the whole window: the window is one part,
        # so the reduction has one value to reduce and the result is that part.
        self.over = over if over is not None else period
        self._check_over()
        self.forecast_items = forecast_items
        self.accumulating = accumulating
        self._logs: Logs | None = None

    def _check_over(self) -> None:
        """Validate the subwindow length against the window.

        The recipe schema checks this too, but a plan can be built directly -- by the
        reductions, by a test, by anything later -- and an unchecked ``over`` fails
        much further on, inside :func:`~.subwindows.validate_partition`, complaining
        about a covering that does not line up rather than about the length that made
        it so.
        """
        if self.over <= datetime.timedelta(0):
            raise ValueError(f"'over' must be positive, got {frequency_to_string(self.over)}")
        if self.over > self.period:
            raise ValueError(
                f"'over' ({frequency_to_string(self.over)}) cannot exceed 'period' "
                f"({frequency_to_string(self.period)}): a subwindow is part of the window."
            )
        if self.period % self.over != datetime.timedelta(0):
            raise ValueError(
                f"'over' ({frequency_to_string(self.over)}) must divide 'period' "
                f"({frequency_to_string(self.period)}) exactly, or the subwindows would "
                "not partition the window."
            )

    def _cover(self, start, end, basetime):
        """One window's subwindows, as the covering resolves them."""
        if self.forecast_items is not None:
            return list(self.covering.partition(start, end, basetime=basetime))
        return list(self.covering.partition(start, end))

    def _partition(self, start, end, basetime) -> list:
        """The window's subwindows: each ``over``-long slice, covered on its own.

        ``over`` defaults to the whole period, so the default runs this loop exactly
        once and asks the covering for ``[start, end]`` -- which is what the search
        has always been asked. For a cumulative archive that is one subwindow and two
        fields: right for a sum, and useless for anything else. A shorter ``over``
        covers each slice separately, so the window comes back as N subwindows each
        rebuilt by differencing.
        """
        subwindows: list = []
        at = start
        while at < end:
            subwindows.extend(self._cover(at, at + self.over, basetime))
            at += self.over
        validate_partition(subwindows, start, end)
        return subwindows

    def parts_for(self, targets: list[Target]) -> dict[Target, list]:
        parts = {}
        for valid_date, basetime in targets:
            covering = self._partition(valid_date - self.period, valid_date, basetime)
            parts[(valid_date, basetime)] = covering
            self._check_reducible(covering, valid_date - self.period, valid_date)
            LOG.debug("  Covering of %s to %s:", valid_date - self.period, valid_date)
            for subwindow in covering:
                LOG.debug("    %s", subwindow)
                for contribution in subwindow.contributions:
                    LOG.debug("       %s", contribution)
        return parts

    def _check_reducible(self, subwindows, start, end) -> None:
        """Reject a covering whose subwindows cannot carry the declared statistic.

        The subwindow invariants are arithmetic about time spans: they accept
        ``-max(0,6) +max(0,9)`` as a reconstruction of ``[6,9]``, because ``-6h + 9h``
        really is ``3h``. Whether differencing *means* anything depends on what the
        archive stores, which no invariant can see.

        ``from:`` and the block name are the only declarations we have, and in the
        ordinary case they agree: ``maximum:`` over a gust archive means the archived
        fields are maxima. So a non-additive reduction needs every subwindow direct --
        unless the recipe asked for subwindows *shorter than the window*, which is it
        declaring that the underlying quantity is additive and stating how long each
        subwindow is.

        Note the test is ``over < period``, not "was ``over:`` written". Since ``over``
        defaults to the period, testing whether it was given would lift this guard on
        every recipe and the check would never fire at all.

        Declaring ``over:`` does not make differencing safe on its own; it makes the
        *quantity* well defined. Whether the archive is really additive is settled when
        a field arrives, by :func:`~.states.field_statistic`.
        """
        # `from:` states the archived statistic whenever it declares an accumulation
        # scheme, so that case is decidable here rather than on the first field. It is
        # independent of the partition and of `over:`, so it comes before both.
        if self.accumulating and not self.operation.reduces_archived("accum"):
            raise ValueError(self.operation.why_not_archived("accum"))

        if self.operation.differenceable or self.over < self.period:
            return

        differenced = [s for s in subwindows if not s.is_direct]

        detail = "\n".join(
            "    {} would be rebuilt from {}".format(
                s, " ".join(f"{'+' if c.sign > 0 else '-'}{c}" for c in s.contributions)
            )
            for s in differenced
        )
        if not differenced:
            self._check_not_degenerate(subwindows, start, end)
            return

        raise ValueError(
            f"{self.operation.name!r} over the window {start} -> {end} needs subwindows the "
            f"archive does not hold outright:\n{detail}\n"
            f"This will not work: {self.operation.why_not_differenceable()}.\n"
            "Achievable windows are unions of whole archived intervals. Check that 'from:' "
            "describes how *this* parameter is stored -- a recognised description describes "
            "precipitation-style layouts, and a parameter such as wind gust is often archived "
            "with different step ranges in the very same class/stream.\n"
            "If this parameter is additive after all -- a cumulative total rather than a "
            f"stored {self.operation.name} -- then say how long each subwindow should be with "
            "'over:', e.g. 'over: 1h' for the largest hourly value in the window."
        )

    def _check_not_degenerate(self, subwindows, start, end) -> None:
        """Reject a non-additive reduction over a single accumulated subwindow.

        The covering guard above catches differencing. This catches the case where
        nothing is differenced and the reduction is still meaningless: the archive
        holds the whole window outright, so the partition is one part, and a max over
        one part is that part. When the part is an *accumulation* the result is the
        window's total, stamped as a maximum.

        It happens on an ordinary archive. `od-oper` stores `a(0,s)` from runs at 00Z
        and 12Z, so a 6 h window starting at a basetime is held outright while the
        next one has to be differenced -- the guard above fires on the second and this
        one on the first, and without both, half the rows of one recipe would come out
        silently wrong.

        Only for an accumulating archive. A gust archive that holds `[0,6]` outright
        really does answer "the maximum over that window" with one field, and it is
        described by explicit step pairs, which carry no ``accumulation``.
        """
        if not (self.accumulating and len(subwindows) == 1):
            return

        raise ValueError(
            f"{self.operation.name!r} over the window {start} -> {end} would reduce a single "
            f"value, because the archive holds that whole window outright as {subwindows[0]}.\n"
            f"A {self.operation.name} over one part is that part, and 'from:' says the part is "
            f"an accumulation -- so the result would be the window's total with a "
            f"{self.operation.name!r} label on it, not a {self.operation.name} of anything.\n"
            "Say how long each subwindow should be with 'over:', e.g. 'over: 1h' for the "
            "largest hourly total in the window."
        )

    def argument(self, targets: list[Target], parts: dict[Target, list]) -> Any:
        # Windows overlap whenever `period` exceeds the output frequency, so the same
        # archived interval is wanted by several of them. Fetch each once: matching
        # remaps it onto every window that needs it.
        intervals = _unique(contributions_of(s for t in targets for s in parts[t]))

        if self.forecast_items is not None:
            return ForecastIntervals(
                items=[(vt, bt, self.period) for vt, bt in self.forecast_items],
                intervals=intervals,
            )
        return Intervals(dates=sorted({vt for vt, _ in targets}), intervals=intervals)

    def new_reducer(self, target: Target, key: tuple, parts: list) -> Reducer:
        valid_date, basetime = target
        return Reducer(
            valid_date,
            period=self.period,
            key=key,
            states=[SubwindowState(subwindow) for subwindow in parts],
            operation=self.operation,
            basetime=basetime if self.basetime else None,
        )

    def identify(self, field: Any) -> Any:
        return self.field_to_interval(field)

    def field_info(self, field: Any) -> str:
        """What statistic the field carries, for the field-time guard."""
        return field_statistic(field)

    def offer(self, reducer: Reducer, values: Any, identity: Any, info: Any) -> bool:
        return reducer.compute(values, identity, statistic=info)

    # ── diagnostics ──────────────────────────────────────────────────

    def begin(self, input_fields: Any, reducers: dict) -> None:
        self._logs = Logs(
            reducers=reducers,
            source=self.source,
            source_object=input_fields,
            field_to_interval=self.field_to_interval,
        )

    def note(self, field: Any, identity: Any, used_by: list) -> None:
        """Record that this field arrived, and which windows took it.

        Runs for every field of every build and is thrown away unless something
        fails, so it records only what cannot be recovered later: the field, what it
        was taken to be, and where it went. It deliberately does *not* snapshot each
        reducer -- one verbose repr per ``(field, window)`` pair is minutes of string
        formatting and gigabytes retained on a large build, and
        :meth:`~.reducer.Logs.raise_error` prints every reducer's state in full
        anyway, at the moment it matters.
        """
        meta = field.get(collections="metadata.mars")
        log = " ".join(f"{k}={v}" for k, v in meta.items())
        self._logs.append([str(field), log, identity, [t for t, _ in used_by]])

    def unused_field(self, field: Any, identity: Any) -> None:
        self._logs.raise_error("Field not used for any accumulation", field=field, field_interval=identity)
