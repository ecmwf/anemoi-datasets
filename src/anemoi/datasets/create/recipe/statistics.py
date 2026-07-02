# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import numpy as np
from anemoi.transform.fields import to_datetime
from pydantic import BaseModel
from pydantic import Field

from anemoi.datasets.usage.misc import as_first_date
from anemoi.datasets.usage.misc import as_last_date

LOG = logging.getLogger(__name__)


class Statistics(BaseModel):
    start: str | int | datetime.datetime | None = None
    end: str | int | datetime.datetime | None = None
    tendencies: list[str | int] | bool | None = Field(default=None)
    """Tendency statistics to compute. ``None`` (default) and ``False`` disable them,
    ``True`` uses the default deltas, a list gives explicit deltas."""

    allow_nans: bool | list[str] | None = Field(
        default=None,
        deprecated="'statistics.allow_nans' is deprecated. Please use 'build.allow_nans' instead.",
    )

    @classmethod
    def default_statistics_dates(cls, dates: list[datetime.datetime]) -> tuple[datetime.datetime, datetime.datetime]:
        """Calculate default statistics dates based on the given list of dates.

        Parameters
        ----------
        dates : list of datetime.datetime
            List of datetime objects representing dates.

        Returns
        -------
        tuple of datetime.datetime
            A tuple containing the default start and end dates.
        """

        first = dates[0]
        last = dates[-1]

        first = to_datetime(first)
        last = to_datetime(last)

        n_years = round((last - first).total_seconds() / (365.25 * 24 * 60 * 60))
        n_dates = len(dates)

        if n_dates < 10:
            # For test datasets.
            k = max(1, int(n_dates * 0.8))
            end = dates[k - 1]
            LOG.info(f"Number of datetimes {n_dates} < 10, leaving out 20%. {end=}")
            return dates[0], end

        if n_years < 10:
            # leave out 20% of the data
            k = int(len(dates) * 0.8)
            end = dates[k - 1]
            LOG.info(f"Number of years {n_years} < 10, leaving out 20%. {end=}")
            return dates[0], end

        delta = 1
        if n_years >= 20:
            delta = 3
        LOG.info(f"Number of years {n_years}, leaving out {delta} years.")
        end_year = last.year - delta

        end = max(d for d in dates if to_datetime(d).year == end_year)
        return dates[0], end

    def statistics_dates(self, dates: list[datetime.datetime]) -> tuple[datetime.datetime, datetime.datetime]:

        start = self.start
        end = self.end
        # if not specified, use the default statistics dates
        default_start, default_end = Statistics.default_statistics_dates(dates)
        if start is None:
            start = default_start
        if end is None:
            end = default_end

        # in any case, adapt to the actual dates in the dataset
        start = as_first_date(start, dates)
        end = as_last_date(end, dates)

        assert start <= end, f"Invalid statistics date range: start={start}, end={end}"
        return start, end

    def statistics_filter(self, dates):
        start, end = self.statistics_dates(dates)
        start = np.datetime64(start).astype(dates[0].dtype)
        end = np.datetime64(end).astype(dates[0].dtype)

        LOG.info(f"Using statistics date range: start={start}, end={end}")

        return StatisticsFilter(start, end)

    def trajectory_statistics_filter(self, base_dates, steps) -> "TrajectoryStatisticsFilter":
        """Build an envelope filter for a trajectory dataset.

        Defaults for ``start``/``end`` (when unset on the recipe) are picked
        from the sorted-unique valid-time list — i.e. every ``base_date +
        step`` pair — so the cut-off reflects the actual time coverage of
        the dataset rather than just its forecast-initialisation times.

        Parameters
        ----------
        base_dates : array-like of np.datetime64
            The base dates of the trajectory dataset.
        steps : array-like of np.timedelta64
            The forecast steps of the trajectory dataset.

        Returns
        -------
        TrajectoryStatisticsFilter
            Filter restricting trajectory pairs to the configured envelope.
        """
        base_dates = np.asarray(base_dates)
        steps = np.asarray(steps)
        assert np.all(np.diff(steps) >= 0), f"steps must be sorted ascending, got {steps}"
        valid_times = np.unique((base_dates[:, None] + steps[None, :]).ravel())

        start, end = self.statistics_dates(list(valid_times))
        np_start = np.datetime64(start).astype(base_dates.dtype)
        np_end = np.datetime64(end).astype(base_dates.dtype)

        LOG.info(f"Using trajectory statistics envelope: start={np_start}, end={np_end}")

        return TrajectoryStatisticsFilter(np_start, np_end, steps[0], steps[-1])


class StatisticsFilter:
    def __init__(self, start, end):
        self.statistics_start_date = start
        self.statistics_end_date = end

    def __call__(self, array, dates, offset=0):
        start = np.searchsorted(dates, self.statistics_start_date, side="left")
        end = np.searchsorted(dates, self.statistics_end_date, side="right")
        start = max(0, min(start, start + offset))
        end = min(len(array), max(end, end + offset))
        return array[start:end]

    def __eq__(self, other):
        if not isinstance(other, StatisticsFilter):
            return False
        return (
            self.statistics_start_date == other.statistics_start_date
            and self.statistics_end_date == other.statistics_end_date
        )


class TrajectoryStatisticsFilter:
    """Envelope filter for trajectory datasets.

    A trajectory (one ``base_date`` row of the 5-D array) is included only
    when its full valid-time envelope ``[base_date + step_start,
    base_date + step_end]`` lies within ``[statistics_start_date,
    statistics_end_date]``.  Otherwise the whole trajectory is dropped, matching
    the behaviour of ``open_dataset(start=, end=)`` for trajectories.

    Parameters
    ----------
    statistics_start_date : np.datetime64
        Inclusive lower bound of the statistics interval (valid-time).
    statistics_end_date : np.datetime64
        Inclusive upper bound of the statistics interval (valid-time).
    step_start : np.timedelta64
        First step of the dataset.
    step_end : np.timedelta64
        Last step of the dataset.
    """

    def __init__(self, statistics_start_date, statistics_end_date, step_start, step_end):
        assert step_start <= step_end, f"step_start must be <= step_end, got {step_start} > {step_end}"
        self.statistics_start_date = statistics_start_date
        self.statistics_end_date = statistics_end_date
        self.step_start = step_start
        self.step_end = step_end
        # Trajectories whose base_date falls in [_base_min, _base_max] have their
        # entire envelope within [statistics_start_date, statistics_end_date].
        self._base_min = statistics_start_date - step_start
        self._base_max = statistics_end_date - step_end

    def mask(self, base_dates):
        """Return a boolean mask over ``base_dates`` selecting kept trajectories."""
        base_dates = np.asarray(base_dates)
        return (base_dates >= self._base_min) & (base_dates <= self._base_max)

    def __eq__(self, other):
        if not isinstance(other, TrajectoryStatisticsFilter):
            return False
        return (
            self.statistics_start_date == other.statistics_start_date
            and self.statistics_end_date == other.statistics_end_date
            and self.step_start == other.step_start
            and self.step_end == other.step_end
        )
