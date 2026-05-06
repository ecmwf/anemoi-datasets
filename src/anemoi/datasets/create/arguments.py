# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Typed argument classes for source dispatch.

The four classes form a 2 × 2 matrix:

                  Instant (one snapshot)    Aggregate (one value per window)
No basetime       ValidDates                Intervals
With basetime     ForecastDates             ForecastIntervals

- ValidDates      — list of validity times (analysis / reanalysis / obs)
- ForecastDates   — list of (valid_time, basetime) pairs (NWP instant forecasts)
- Intervals       — archive-resolved accumulation windows; subclass of ValidDates.
                    Replaces IntervalsDatesProvider.  Each date maps to a list of
                    SignedInterval objects from covering_intervals.py.
- ForecastIntervals — list of (valid_time, basetime, period) triples;
                    subclass of ForecastDates.

Inheritance:
    Intervals        ← ValidDates      (MRO fallback: sources that only register
    ForecastIntervals ← ForecastDates   execute_valid_dates / execute_forecast_dates
                                        automatically handle the interval subtypes)

Conversion helpers on each class let composite sources transform the caller's
request before passing it to an inner source:

    valid_dates.as_intervals(period)          → Intervals  (trivial coverage)
    valid_dates.with_basetime(fn)             → ForecastDates
    forecast_dates.as_forecast_intervals(p)   → ForecastIntervals
    intervals.with_basetime(fn)               → ForecastIntervals  (not implemented)
"""

from __future__ import annotations

import datetime
from typing import Any
from typing import Callable

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Argument:
    """Base class for all typed source arguments."""


# ---------------------------------------------------------------------------
# ValidDates
# ---------------------------------------------------------------------------


class ValidDates(Argument):
    """A list of validity times for instant analysis / reanalysis / observation data.

    Parameters
    ----------
    dates : list[datetime.datetime]
        The validity times requested by the pipeline.
    """

    def __init__(self, dates: list[datetime.datetime]) -> None:
        self.dates = list(dates)

    # -- Conversion helpers --------------------------------------------------

    def as_intervals(self, period: datetime.timedelta) -> "Intervals":
        """Promote to Intervals with a trivial single-interval coverage.

        Each date T becomes one SignedInterval covering ``[T − period, T]``.
        This is a *geometric* conversion that carries no archive-availability
        knowledge.  AccumulateSource uses ``interval_generator_factory`` for
        the real covering-interval decomposition instead.

        Parameters
        ----------
        period : datetime.timedelta
            The accumulation window length.
        """
        from anemoi.datasets.create.intervals import SignedInterval

        return Intervals(self.dates, [SignedInterval(d - period, d) for d in self.dates])

    def with_basetime(self, basetime_of: Callable[[datetime.datetime], datetime.datetime]) -> "ForecastDates":
        """Attach a model-run time to each instant.

        Parameters
        ----------
        basetime_of : Callable[[datetime], datetime]
            Pure function mapping a validity time to its base time
            (e.g. rounding to the nearest 00/12 UTC run).
        """
        return ForecastDates([(t, basetime_of(t)) for t in self.dates])

    # -- List-like interface for backward compatibility ----------------------

    def __len__(self) -> int:
        return len(self.dates)

    def __iter__(self):
        return iter(self.dates)

    def __getitem__(self, index):
        return self.dates[index]

    def __repr__(self) -> str:
        return f"ValidDates({len(self.dates)} dates)"


# ---------------------------------------------------------------------------
# ForecastDates
# ---------------------------------------------------------------------------


class ForecastDates(Argument):
    """A list of (valid_time, basetime) pairs for instantaneous NWP forecasts.

    Parameters
    ----------
    items : list[tuple[datetime, datetime]]
        Each entry is ``(valid_time, basetime)`` where basetime is the
        model-run time and ``step = valid_time − basetime``.
    """

    def __init__(self, items: list[tuple[datetime.datetime, datetime.datetime]]) -> None:
        self.items = list(items)

    @property
    def valid_dates(self) -> ValidDates:
        """The validity times as a ValidDates object."""
        return ValidDates([vd for vd, _ in self.items])

    def as_forecast_intervals(self, period: datetime.timedelta) -> "ForecastIntervals":
        """Promote each (valid_time, basetime) to (valid_time, basetime, period).

        Parameters
        ----------
        period : datetime.timedelta
            The accumulation window length.
        """
        return ForecastIntervals([(vd, bt, period) for vd, bt in self.items])

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self) -> str:
        return f"ForecastDates({len(self.items)} items)"


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


class Intervals(ValidDates):
    """Archive-resolved accumulation windows.

    Subclass of ``ValidDates`` so that sources registering only
    ``execute_valid_dates`` receive ``Intervals`` via the default fallback.  Sources that
    care about the period (e.g. ``MarsSource``) override ``execute_intervals``
    explicitly to build step-range requests.

    Each ``SignedInterval`` carries its own ``base_time`` (the model-run
    time, ``None`` only for grib_index) and ``valid_time`` (== ``max``),
    so consumers don't need a parallel index from validity time to
    intervals.

    Parameters
    ----------
    dates : list[datetime.datetime]
        The validity times (output dates) in order.
    intervals : list[SignedInterval]
        Flat list of all archive intervals needed across every output
        date. Order is the iteration order seen by source consumers.
    """

    def __init__(
        self,
        dates: list[datetime.datetime],
        intervals: list,
    ) -> None:
        super().__init__(dates)
        self._intervals = list(intervals)

    @property
    def intervals(self):
        """The flat list of ``SignedInterval`` objects."""
        return self._intervals

    def adjust_request(self, interval: Any, request: dict) -> tuple[Any, dict, int]:
        """Adjust a request dict to reflect a specific SignedInterval.

        Rewrites the request as a full ``date``/``time``/``step`` triplet
        anchored on the interval's model-run time. ``interval.base`` must be
        set; valid-time-indexed backends (grib_index) handle ``base=None``
        intervals in their own ``execute_intervals`` override rather than going
        through this helper.

        Parameters
        ----------
        interval : SignedInterval
            The archive interval (``base`` must not be ``None``).
        request : dict
            Base request to adjust (copied, not mutated).

        Returns
        -------
        tuple
            ``(valid_time, adjusted_request, step_hours)``
        """
        assert interval.base is not None, (
            f"Intervals.adjust_request requires a basetime; got {interval!r}. "
            "Valid-time-indexed sources (e.g. grib_index) must override execute_intervals "
            "and not call this helper."
        )
        r = request.copy()
        step = int((interval.max - interval.base).total_seconds() / 3600)
        r["date"] = interval.base.strftime("%Y%m%d")
        r["time"] = interval.base.strftime("%H%M")
        r["step"] = step
        return interval.max, r, step

    def with_basetime(self, basetime_of: Callable[[datetime.datetime], datetime.datetime]) -> "ForecastIntervals":
        """Attach a model-run time to each window.

        Parameters
        ----------
        basetime_of : Callable[[datetime], datetime]
            Pure function mapping a validity time to its base time.
        """
        raise NotImplementedError(
            "with_basetime on archive-resolved Intervals is not supported. "
            "Use AccumulateSource with ForecastDates input instead."
        )

    def __repr__(self) -> str:
        return f"Intervals({len(self.dates)} dates, {len(self._intervals)} intervals)"


# ---------------------------------------------------------------------------
# ForecastIntervals
# ---------------------------------------------------------------------------


class ForecastIntervals(ForecastDates):
    """Forecast accumulations: ``(valid_time, basetime, period)`` items plus a flat list of intervals.

    Subclass of ``ForecastDates`` so that sources registering only
    ``execute_forecast_dates`` receive ``ForecastIntervals`` via the default fallback.
    Sources that care about the period override ``execute_forecast_intervals``
    explicitly.

    Each ``SignedInterval`` already carries its ``base_time`` (the
    basetime imposed by the caller) and ``valid_time``, so consumers
    don't need a parallel ``(vt, bt) → intervals`` index.

    Parameters
    ----------
    items : list[tuple[datetime, datetime, timedelta]]
        Each entry is ``(valid_time, basetime, period)``.
    intervals : list[SignedInterval], optional
        Flat list of archive intervals across every output item.
        Populated by sources that own the covering (e.g.
        ``AccumulateSource``); empty by default.
    """

    def __init__(
        self,
        items: list[tuple[datetime.datetime, datetime.datetime, datetime.timedelta]],
        intervals: list | None = None,
    ) -> None:
        # ForecastDates.__init__ expects (vt, bt) pairs.
        super().__init__([(vt, bt) for vt, bt, _ in items])
        self.items = list(items)
        self._intervals: list = list(intervals or [])

    @property
    def valid_dates(self) -> ValidDates:
        """The validity times as a ValidDates object."""
        return ValidDates([vd for vd, _, _ in self.items])

    @property
    def forecast_dates(self) -> ForecastDates:
        """The (valid_time, basetime) pairs without period."""
        return ForecastDates([(vd, bt) for vd, bt, _ in self.items])

    @property
    def intervals(self):
        """The flat list of ``SignedInterval`` objects."""
        return self._intervals

    def adjust_request(self, interval: Any, request: dict) -> tuple[Any, dict, int]:
        """Adjust a request dict to reflect a specific SignedInterval.

        Rewrites the request as a full ``date``/``time``/``step`` triplet
        anchored on the interval's model-run time.

        Parameters
        ----------
        interval : SignedInterval
            The archive interval (``base`` set by ``ForecastCovering``).
        request : dict
            Base MARS/FDB request to adjust (copied, not mutated).

        Returns
        -------
        tuple
            ``(valid_time, adjusted_request, step_hours)``
        """
        r = request.copy()
        step = int((interval.max - interval.base).total_seconds() / 3600)
        r["date"] = interval.base.strftime("%Y%m%d")
        r["time"] = interval.base.strftime("%H%M")
        r["step"] = step
        return interval.max, r, step

    def __repr__(self) -> str:
        return f"ForecastIntervals({len(self.items)} items, {len(self._intervals)} intervals)"
