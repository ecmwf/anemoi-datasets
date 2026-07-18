# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Any

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry

from .retrieval import RequestFilter
from .retrieval import execute_mars_request
from .retrieval import fire_prebuilt_requests


def _hindcast_refdate_mapping(config: dict[str, Any]) -> dict[datetime.date, datetime.date]:
    """Build the hdate → refdate lookup table for the mars ``hindcast`` option.

    Reforecast (hindcast) MARS requests need both ``date`` (the reference
    date of the model run) and ``hdate`` (the start of the hindcast run,
    same month-day in an earlier year). Given the reference-date window,
    every hdate produced by the archive convention (``years`` previous
    years of each reference date) is mapped back to its reference date.

    Parameters
    ----------
    config : dict
        The ``hindcast`` option: ``reference_start``/``reference_end``
        (the reference-date window), optional ``day_of_week`` (e.g.
        ``[monday, thursday]``) restricting the reference dates, and
        optional ``years`` (number of hindcast years, default 20).

    Returns
    -------
    dict
        Mapping of hdate to reference date.
    """
    from anemoi.utils.dates import DateTimes
    from anemoi.utils.hindcasts import HindcastDatesTimes

    config = dict(config)
    start = config.pop("reference_start")
    end = config.pop("reference_end")
    day_of_week = config.pop("day_of_week", None)
    years = config.pop("years", 20)
    if config:
        raise ValueError(f"mars hindcast: unknown option(s) {sorted(config)}")

    kwargs = {}
    if day_of_week is not None:
        if isinstance(day_of_week, str):
            day_of_week = [day_of_week]
        kwargs["day_of_week"] = [d.lower() if isinstance(d, str) else d for d in day_of_week]

    reference_dates = list(DateTimes(start, end, increment=24, **kwargs))
    if not reference_dates:
        raise ValueError(f"mars hindcast: no reference dates in [{start}, {end}] (day_of_week={day_of_week})")

    mapping: dict[datetime.date, datetime.date] = {}
    for hdate, refdate in HindcastDatesTimes(reference_dates=reference_dates, years=years):
        key = hdate.date()
        other = mapping.get(key)
        if other is not None and other != refdate.date():
            raise ValueError(
                f"mars hindcast: hdate {key} is a hindcast date of two reference dates "
                f"({other} and {refdate.date()}); narrow the [reference_start, reference_end] "
                "window or set day_of_week to make the mapping unique"
            )
        mapping[key] = refdate.date()
    return mapping


def _reject_filters(requests: list[dict[str, Any]], context_label: str) -> None:
    """Raise if any request carries a per-step filter (wildcard ``date``).
    Filters only make sense in the validity-date path: the forecast-date
    and interval paths own their own date/time/step arithmetic, so a filter
    would be silently overwritten or incoherent.
    """
    for r in requests:
        filter_, _ = RequestFilter.extract(r)
        if not filter_.is_empty:
            raise ValueError(f"Wildcard 'date' filters are not supported in " f"{context_label} mars blocks.")


# TODO: there is some code duplication between here and FDB source, might be reduced
@source_registry.register("mars")
class MarsSource(Source):

    def __init__(self, context: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(context, *args, **kwargs)
        self.use_cdsapi_dataset = kwargs.pop("use_cdsapi_dataset", None)
        self.hindcast = kwargs.pop("hindcast", None)
        self._hindcast_refdates = None if self.hindcast is None else _hindcast_refdate_mapping(self.hindcast)
        self.args = args
        self.kwargs = kwargs

    def _apply_hindcast(self, request: dict[str, Any], basetime: datetime.datetime) -> dict[str, Any]:
        """Rewrite a stamped forecast request as a reforecast (hindcast) request.

        The incoming request carries ``date``/``time`` from the basetime and a
        ``step`` relative to it. For hindcast streams (eefh/enfh) the basetime
        is the hindcast start (``hdate``, always 00Z) while ``date`` must be the
        reference date of the model run; the step stays relative to the hdate.

        Parameters
        ----------
        request : dict
            The request to rewrite (mutated in place).
        basetime : datetime.datetime
            The model-run basetime of the trajectory item.
        """
        if (basetime.hour, basetime.minute, basetime.second) != (0, 0, 0):
            raise ValueError(f"mars hindcast: basetime must be at 00Z, got {basetime}")
        refdate = self._hindcast_refdates.get(basetime.date())
        if refdate is None:
            raise ValueError(
                f"mars hindcast: basetime {basetime.date()} is not a hindcast date of any "
                f"reference date in [{self.hindcast['reference_start']}, {self.hindcast['reference_end']}] "
                f"(day_of_week={self.hindcast.get('day_of_week')}, years={self.hindcast.get('years', 20)})"
            )
        request["hdate"] = basetime.strftime("%Y%m%d")
        request["date"] = refdate.strftime("%Y%m%d")
        request["time"] = "0000"
        return request

    def execute_valid_dates(self, dates: ValidDates) -> Any:
        """Handle instant analysis / reanalysis requests."""
        if self.hindcast is not None:
            raise ValueError(
                "The mars 'hindcast' option needs a basetime for each date and is only "
                "supported in forecast contexts (trajectories layout), not with validity dates."
            )
        if not dates.dates:
            # No validity dates: the request already encodes its own date
            # (e.g. repeated_dates constant mode with date=None).
            # Route directly through fire_prebuilt_requests instead of going
            # through execute_mars_request with an empty date list.
            requests = list(self.args) or [self.kwargs.copy()]
            for r in requests:
                if isinstance(r.get("date"), datetime.date):
                    r["date"] = r["date"].strftime("%Y%m%d")
            return fire_prebuilt_requests(self.context, requests, self.use_cdsapi_dataset)
        return execute_mars_request(
            self.context, dates.dates, *self.args, use_cdsapi_dataset=self.use_cdsapi_dataset, **self.kwargs
        )

    def execute_forecast_dates(self, dates: ForecastDates) -> Any:
        """Handle forecast (basetime, valid_time) requests — trajectories / step products."""
        base_requests = list(self.args) or [self.kwargs]
        _reject_filters(base_requests, "forecast-date")
        per_item_requests: list[dict[str, Any]] = []
        for valid_time, basetime in dates.items:
            step_seconds = (valid_time - basetime).total_seconds()
            if step_seconds % 3600:
                raise ValueError(
                    f"MARS forecast requests only support whole-hour steps, got "
                    f"step={valid_time - basetime} (valid_time={valid_time}, basetime={basetime})."
                )
            step_hours = int(step_seconds // 3600)
            for request in base_requests:
                r = request.copy()
                r["date"] = basetime.strftime("%Y%m%d")
                r["time"] = basetime.strftime("%H%M")
                r["step"] = step_hours
                if self._hindcast_refdates is not None:
                    self._apply_hindcast(r, basetime)
                per_item_requests.append(r)

        self.context.trace("🛰️", f"Forecast dates: {len(dates)} items → {len(per_item_requests)} requests")
        return fire_prebuilt_requests(self.context, per_item_requests, self.use_cdsapi_dataset)

    def execute_intervals(self, dates: Intervals) -> Any:
        """Handle archive-resolved interval requests from AccumulateSource."""
        if self.hindcast is not None:
            raise ValueError(
                "The mars 'hindcast' option needs a basetime for each date and is only "
                "supported in forecast contexts (trajectories layout), not with validity dates."
            )
        base_requests = list(self.args) or [self.kwargs]
        _reject_filters(base_requests, "accumulation")
        per_interval_requests: list[dict[str, Any]] = []
        for request in base_requests:
            for interval in dates.intervals:
                # MARS sources always have a model-run time; only grib_index is
                # allowed to produce base-less intervals (flat valid-time index).
                assert interval.base is not None, (
                    f"MarsSource received an interval without a basetime: {interval!r}. "
                    "Only grib_index is expected to produce base=None intervals."
                )
                self.context.trace("🌧️", "interval:", interval)
                _, r, _ = dates.adjust_request(interval, request)
                self.context.trace("🌧️", "  adjusted request =", r)
                per_interval_requests.append(r)

        self.context.trace("🌧️", f"Total requests: {len(per_interval_requests)}")
        return fire_prebuilt_requests(self.context, per_interval_requests, self.use_cdsapi_dataset)

    def execute_forecast_intervals(self, dates: ForecastIntervals) -> Any:
        """Handle forecast (trajectory) accumulation requests.

        One MARS request per ``(valid_time, basetime, SignedInterval)``
        triple. Date/time are stamped from the basetime; step is the
        offset from basetime to the interval endpoint.
        """
        base_requests = list(self.args) or [self.kwargs]
        _reject_filters(base_requests, "forecast accumulation")
        per_interval_requests: list[dict[str, Any]] = []
        for request in base_requests:
            for interval in dates.intervals:
                # Trajectory accumulations always go through ForecastCovering, which
                # sets base=basetime; a base-less interval here would be a bug.
                assert interval.base is not None, (
                    f"MarsSource received a forecast interval without a basetime: {interval!r}. "
                    "Only grib_index is expected to produce base=None intervals."
                )
                self.context.trace(
                    "\U0001f327\ufe0f",
                    "forecast interval:",
                    interval,
                    "vt=",
                    interval.valid_time,
                    "bt=",
                    interval.base,
                )
                _, r, _ = dates.adjust_request(interval, request)
                if self._hindcast_refdates is not None:
                    self._apply_hindcast(r, interval.base)
                self.context.trace("🌧️", "  adjusted request =", r)
                per_interval_requests.append(r)

        self.context.trace("🌧️", f"Total forecast accumulation requests: {len(per_interval_requests)}")
        return fire_prebuilt_requests(self.context, per_interval_requests, self.use_cdsapi_dataset)
