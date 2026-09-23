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
from ..windowed.reducer import Logs
from ..windowed.reducer import Reducer
from ..windowed.states import SubwindowState
from ..windowed.states import field_statistic
from ..windowed.subwindows import contributions_of

LOG = logging.getLogger(__name__)


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
    basetime_from_target : bool
        Whether the reducer should stamp its output with the target's basetime
        (a trajectory row) rather than the start of the window.
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
        forecast_items: list | None = None,
    ) -> None:
        self.period = period
        self.covering = covering
        self.source = source
        self.field_to_interval = field_to_interval
        self.basetime = basetime
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
            LOG.debug("  Covering of %s to %s:", valid_date - self.period, valid_date)
            for subwindow in covering:
                LOG.debug("    %s", subwindow)
                for contribution in subwindow.contributions:
                    LOG.debug("       %s", contribution)
        return parts

    def argument(self, targets: list[Target], parts: dict[Target, list]) -> Any:
        if self.forecast_items is not None:
            return ForecastIntervals(
                items=[(vt, bt, self.period) for vt, bt in self.forecast_items],
                intervals=contributions_of(s for t in targets for s in parts[t]),
            )

        # Overlapping rows can request the same subsource window more than once; fetch
        # each interval once, since matching remaps it onto every row that needs it.
        seen: dict = {}
        for target in targets:
            for interval in contributions_of(parts[target]):
                seen.setdefault(interval, None)
        return Intervals(dates=sorted({vt for vt, _ in targets}), intervals=list(seen))

    def new_reducer(self, target: Target, key: tuple, parts: list) -> Reducer:
        valid_date, basetime = target
        return Reducer(
            valid_date,
            period=self.period,
            key=key,
            states=[SubwindowState(subwindow) for subwindow in parts],
            operation="sum",
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
