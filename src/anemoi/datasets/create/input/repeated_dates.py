# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict
from collections.abc import Generator
from typing import Any

import numpy as np
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_timedelta

LOG = logging.getLogger(__name__)


class DateMapper:
    """A factory class to create DateMapper instances based on the given mode."""

    @staticmethod
    def from_mode(mode: str, source: Any, config: dict[str, Any]) -> "DateMapper":
        """Create a DateMapper instance based on the given mode.

        Parameters
        ----------
        mode : str
            The mode to use for the DateMapper.
        source : Any
            The data source.
        config : dict
            Configuration parameters.

        Returns
        -------
        DateMapper
            An instance of DateMapper.
        """
        MODES: dict = dict(
            closest=DateMapperClosest,
            climatology=DateMapperClimatology,
            constant=DateMapperConstant,
        )

        if mode not in MODES:
            raise ValueError(f"Invalid mode for DateMapper: {mode}")

        return MODES[mode](source, **config)


class DateMapperClosest(DateMapper):
    """A DateMapper implementation that maps dates to the closest available dates."""

    def __init__(self, source: Any, frequency: str = "1h", maximum: str = "30d", skip_all_nans: bool = False) -> None:
        """Initialize DateMapperClosest.

        Parameters
        ----------
        source : Any
            The data source.
        frequency : str
            Frequency of the dates.
        maximum : str
            Maximum time delta.
        skip_all_nans : bool
            Whether to skip all NaN values.
        """
        self.source: Any = source
        self.maximum: Any = frequency_to_timedelta(maximum)
        self.frequency: Any = frequency_to_timedelta(frequency)
        self.skip_all_nans: bool = skip_all_nans
        self.tried: set[Any] = set()
        self.found: set[Any] = set()

    def transform(self, group_of_dates: Any) -> Generator[tuple[Any, Any], None, None]:
        """Transform the group of dates to the closest available dates.

        Parameters
        ----------
        group_of_dates : Any
            The group of dates to transform.

        Returns
        -------
        Generator[Tuple[Any, Any], None, None]
            Transformed dates.
        """
        from anemoi.datasets.dates.groups import GroupOfDates

        asked_dates = list(group_of_dates)
        if not asked_dates:
            return []

        to_try = set()
        for date in asked_dates:
            start = date
            while start >= date - self.maximum:
                to_try.add(start)
                start -= self.frequency

            end = date
            while end <= date + self.maximum:
                to_try.add(end)
                end += self.frequency

        to_try = sorted(to_try - self.tried)
        info = {k: "no-data" for k in to_try}

        if not to_try:
            LOG.warning(f"No new dates to try for {group_of_dates} in {self.source}")
            # return []

        if to_try:
            result = self.source.select(
                GroupOfDates(
                    sorted(to_try),
                    group_of_dates.provider,
                    partial_ok=True,
                )
            )

            cnt = 0
            for f in result.datasource:
                cnt += 1
                # We could keep the fields in a dictionary, but we don't want to keep the fields in memory
                date = as_datetime(f.metadata("valid_datetime"))

                if self.skip_all_nans:
                    if np.isnan(f.to_numpy()).all():
                        LOG.warning(f"Skipping {date} because all values are NaN")
                        info[date] = "all-nans"
                        continue

                info[date] = "ok"
                self.found.add(date)

            if cnt == 0:
                raise ValueError(f"No data found for {group_of_dates} in {self.source}")

            self.tried.update(to_try)

        if not self.found:
            for k, v in info.items():
                LOG.warning(f"{k}: {v}")

            raise ValueError(f"No matching data found for {asked_dates} in {self.source}")

        new_dates = defaultdict(list)

        for date in asked_dates:
            best = None
            for found_date in sorted(self.found):
                delta = abs(date - found_date)
                # With < we prefer the first date
                # With <= we prefer the last date
                if best is None or delta <= best[0]:
                    best = delta, found_date
            new_dates[best[1]].append(date)

        for date, dates in new_dates.items():
            yield (
                GroupOfDates([date], group_of_dates.provider),
                GroupOfDates(dates, group_of_dates.provider),
            )


class DateMapperClimatology(DateMapper):
    """A DateMapper implementation that maps dates to specified climatology dates."""

    def __init__(self, source: Any, year: int, day: int, hour: int | None = None) -> None:
        """Initialize DateMapperClimatology.

        Parameters
        ----------
        source : Any
            The data source.
        year : int
            The year to map to.
        day : int
            The day to map to.
        hour : Optional[int]
            The hour to map to.
        """
        self.year: int = year
        self.day: int = day
        self.hour: int | None = hour

    def transform(self, group_of_dates: Any) -> Generator[tuple[Any, Any], None, None]:
        """Transform the group of dates to the specified climatology dates.

        Parameters
        ----------
        group_of_dates : Any
            The group of dates to transform.

        Returns
        -------
        Generator[Tuple[Any, Any], None, None]
            Transformed dates.
        """
        from anemoi.datasets.dates.groups import GroupOfDates

        dates = list(group_of_dates)
        if not dates:
            return []

        new_dates = defaultdict(list)
        for date in dates:
            new_date = date.replace(year=self.year, day=self.day)
            if self.hour is not None:
                new_date = new_date.replace(hour=self.hour, minute=0, second=0)
            new_dates[new_date].append(date)

        for date, dates in new_dates.items():
            yield (
                GroupOfDates([date], group_of_dates.provider),
                GroupOfDates(dates, group_of_dates.provider),
            )


class DateMapperConstant(DateMapper):
    """A DateMapper implementation that maps dates to a constant date."""

    def __init__(self, source: Any, date: Any | None = None) -> None:
        """Initialize DateMapperConstant.

        Parameters
        ----------
        source : Any
            The data source.
        date : Optional[Any]
            The constant date to map to.
        """
        self.source: Any = source
        self.date: Any | None = date

    def transform(self, group_of_dates: Any) -> tuple[Any, Any]:
        """Transform the group of dates to a constant date.

        Parameters
        ----------
        group_of_dates : Any
            The group of dates to transform.

        Returns
        -------
        Tuple[Any, Any]
            Transformed dates.
        """
        from anemoi.datasets.dates.groups import GroupOfDates

        if self.date is None:
            return [
                (
                    GroupOfDates([], group_of_dates.provider),
                    group_of_dates,
                )
            ]

        return [
            (
                GroupOfDates([self.date], group_of_dates.provider),
                group_of_dates,
            )
        ]
