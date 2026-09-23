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

That is also why this plan carries :class:`~..windowed.reducer.Logs`: when a field fits
no window there is nothing to point at, so `accumulate` keeps a running account of what
arrived and where it went.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals

from ..windowed.plan import Target
from ..windowed.plan import WindowPlan
from ..windowed.operations import Operation
from ..windowed.operations import operation_factory
from ..windowed.reducer import Logs
from ..windowed.reducer import Reducer
from ..windowed.states import SubwindowState
from ..windowed.states import field_statistic
from ..windowed.subwindows import contributions_of

LOG = logging.getLogger(__name__)


def _unique(intervals) -> list:
    """The intervals, in order, with repeats dropped."""
    seen: dict = {}
    for interval in intervals:
        seen.setdefault(interval, None)
    return list(seen)


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
    forecast_items : list, optional
        ``(valid_time, basetime)`` rows, when the subsource is the run the layout
        imposes; makes the argument a ``ForecastIntervals``.
    """

    def __init__(
        self,
        period: datetime.timedelta,
        covering: Any,
        source: dict,
        field_to_interval: Any,
        basetime: bool = False,
        operation: Operation | str | None = None,
        forecast_items: list | None = None,
    ) -> None:
        self.period = period
        self.covering = covering
        self.source = source
        self.field_to_interval = field_to_interval
        self.basetime = basetime
        self.operation = operation_factory(operation)
        self.forecast_items = forecast_items
        self._logs: Logs | None = None
        self._statistic: str | None = None

    def parts_for(self, targets: list[Target]) -> dict[Target, list]:
        parts = {}
        for valid_date, basetime in targets:
            if self.forecast_items is not None:
                covering = self.covering.partition(valid_date - self.period, valid_date, basetime=basetime)
            else:
                covering = self.covering.partition(valid_date - self.period, valid_date)
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
        fields are maxima. So a non-additive reduction needs every subwindow direct.
        """
        if self.operation.differenceable:
            return

        differenced = [s for s in subwindows if not s.is_direct]
        if not differenced:
            return

        detail = "\n".join(
            "    {} would be rebuilt from {}".format(
                s, " ".join(f"{'+' if c.sign > 0 else '-'}{c}" for c in s.contributions)
            )
            for s in differenced
        )
        raise ValueError(
            f"{self.operation.name!r} over the window {start} -> {end} needs subwindows the "
            f"archive does not hold outright:\n{detail}\n"
            f"This will not work: {self.operation.why_not_differenceable()}.\n"
            "Achievable windows are unions of whole archived intervals. Check that 'from:' "
            "describes how *this* parameter is stored -- a recognised description describes "
            "precipitation-style layouts, and a parameter such as wind gust is often archived "
            "with different step ranges in the very same class/stream."
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

    def offer(self, reducer: Reducer, values: Any, identity: Any) -> bool:
        return reducer.compute(values, identity, statistic=self._statistic)

    def identify(self, field: Any) -> Any:
        # cached per field, so the statistic is read once rather than per target
        self._statistic = field_statistic(field)
        return self.field_to_interval(field)

    # ── diagnostics ──────────────────────────────────────────────────

    def begin(self, input_fields: Any, reducers: dict) -> None:
        self._logs = Logs(
            reducers=reducers,
            source=self.source,
            source_object=input_fields,
            field_to_interval=self.field_to_interval,
        )

    def note(self, field: Any, identity: Any, used_by: list) -> None:
        meta = field.get(collections="metadata.mars")
        log = " ".join(f"{k}={v}" for k, v in meta.items())
        self._logs.append(
            [
                str(field),
                log,
                identity,
                [t for t, _ in used_by],
                [r.__repr__(verbose=True) for _, r in used_by],
            ]
        )

    def unused_field(self, field: Any, identity: Any) -> None:
        self._logs.raise_error("Field not used for any accumulation", field=field, field_interval=identity)
