# (C) Copyright 2024 Anemoi contributors.
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
        self.args = args
        self.kwargs = kwargs

    def execute_valid_dates(self, dates: ValidDates) -> Any:
        """Handle instant analysis / reanalysis requests."""
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
            step_hours = int((valid_time - basetime).total_seconds() // 3600)
            for request in base_requests:
                r = request.copy()
                r["date"] = basetime.strftime("%Y%m%d")
                r["time"] = basetime.strftime("%H%M")
                r["step"] = step_hours
                per_item_requests.append(r)

        self.context.trace("🛰️", f"Forecast dates: {len(dates)} items → {len(per_item_requests)} requests")
        return fire_prebuilt_requests(self.context, per_item_requests, self.use_cdsapi_dataset)

    def execute_intervals(self, dates: Intervals) -> Any:
        """Handle archive-resolved interval requests from AccumulateSource."""
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
                self.context.trace("🌧️", "  adjusted request =", r)
                per_interval_requests.append(r)

        self.context.trace("🌧️", f"Total forecast accumulation requests: {len(per_interval_requests)}")
        return fire_prebuilt_requests(self.context, per_interval_requests, self.use_cdsapi_dataset)
