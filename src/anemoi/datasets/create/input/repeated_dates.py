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
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np
from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_timedelta

from .action import Action
from .action import action_factory
from .join import JoinResult
from .result import Result
from .trace import trace_select

LOG = logging.getLogger(__name__)


class DateMapper:
    """A factory class to create DateMapper instances based on the given mode."""

    @staticmethod
    def from_mode(mode: str, source: Any, config: Dict[str, Any]) -> "DateMapper":
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
        self.tried: Set[Any] = set()
        self.found: Set[Any] = set()

    def transform(self, group_of_dates: Any) -> Generator[Tuple[Any, Any], None, None]:
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

    def __init__(self, source: Any, year: int, day: int, hour: Optional[int] = None) -> None:
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
        self.hour: Optional[int] = hour

    def transform(self, group_of_dates: Any) -> Generator[Tuple[Any, Any], None, None]:
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

    def __init__(self, source: Any, date: Optional[Any] = None) -> None:
        """Initialize DateMapperConstant.

        Parameters
        ----------
        source : Any
            The data source.
        date : Optional[Any]
            The constant date to map to.
        """
        self.source: Any = source
        self.date: Optional[Any] = date

    def transform(self, group_of_dates: Any) -> Tuple[Any, Any]:
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


class DateMapperResult(Result):
    """A Result implementation that updates the valid datetime of the datasource."""

    def __init__(
        self,
        context: Any,
        action_path: List[str],
        group_of_dates: Any,
        source_result: Any,
        mapper: DateMapper,
        original_group_of_dates: Any,
    ) -> None:
        """Initialize DateMapperResult.

        Parameters
        ----------
        context : Any
            The context.
        action_path : list of str
            The action path.
        group_of_dates : Any
            The group of dates.
        source_result : Any
            The source result.
        mapper : DateMapper
            The date mapper.
        original_group_of_dates : Any
            The original group of dates.
        """
        super().__init__(context, action_path, group_of_dates)

        self.source_results: Any = source_result
        self.mapper: DateMapper = mapper
        self.original_group_of_dates: Any = original_group_of_dates

    @property
    def datasource(self) -> Any:
        """Get the datasource with updated valid datetime."""
        result: list = []

        for field in self.source_results.datasource:
            for date in self.original_group_of_dates:
                result.append(new_field_with_valid_datetime(field, date))

        if not result:
            raise ValueError("repeated_dates: no input data found")

        return new_fieldlist_from_list(result)


class RepeatedDatesAction(Action):
    """An Action implementation that selects and transforms a group of dates."""

    def __init__(self, context: Any, action_path: List[str], source: Any, mode: str, **kwargs: Any) -> None:
        """Initialize RepeatedDatesAction.

        Args:
            context (Any): The context.
            action_path (List[str]): The action path.
            source (Any): The data source.
            mode (str): The mode for date mapping.
            **kwargs (Any): Additional arguments.
        """
        super().__init__(context, action_path, source, mode, **kwargs)

        self.source: Any = action_factory(source, context, action_path + ["source"])
        self.mapper: DateMapper = DateMapper.from_mode(mode, self.source, kwargs)

    @trace_select
    def select(self, group_of_dates: Any) -> JoinResult:
        """Select and transform the group of dates.

        Args:
            group_of_dates (Any): The group of dates to select.

        Returns
        -------
        JoinResult
            The result of the join operation.
        """
        results: list = []
        for one_date_group, many_dates_group in self.mapper.transform(group_of_dates):
            results.append(
                DateMapperResult(
                    self.context,
                    self.action_path,
                    one_date_group,
                    self.source.select(one_date_group),
                    self.mapper,
                    many_dates_group,
                )
            )

        return JoinResult(self.context, self.action_path, group_of_dates, results)

    def __repr__(self) -> str:
        """Get the string representation of the action.

        Returns
        -------
        str
            The string representation.
        """
        return f"MultiDateMatchAction({self.source}, {self.mapper})"
