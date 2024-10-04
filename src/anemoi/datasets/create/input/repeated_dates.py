# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from collections import defaultdict

import numpy as np
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.fields import FieldArray
from anemoi.datasets.fields import NewValidDateTimeField

from .action import Action
from .action import action_factory
from .join import JoinResult
from .result import Result
from .trace import trace_select

LOG = logging.getLogger(__name__)


class DateMapper:

    @staticmethod
    def from_mode(mode, source, config):

        MODES = dict(
            closest=DateMapperClosest,
            climatology=DateMapperClimatology,
            constant=DateMapperConstant,
        )

        if mode not in MODES:
            raise ValueError(f"Invalid mode for DateMapper: {mode}")

        return MODES[mode](source, **config)


class DateMapperClosest(DateMapper):
    def __init__(self, source, frequency="1h", maximum="30d", skip_all_nans=False):
        self.source = source
        self.maximum = frequency_to_timedelta(maximum)
        self.frequency = frequency_to_timedelta(frequency)
        self.skip_all_nans = skip_all_nans
        self.tried = set()
        self.found = set()

    def transform(self, group_of_dates):
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

        if to_try:
            result = self.source.select(
                GroupOfDates(
                    sorted(to_try),
                    group_of_dates.provider,
                    partial_ok=True,
                )
            )

            for f in result.datasource:
                # We could keep the fields in a dictionary, but we don't want to keep the fields in memory
                date = as_datetime(f.metadata("valid_datetime"))

                if self.skip_all_nans:
                    if np.isnan(f.to_numpy()).all():
                        LOG.warning(f"Skipping {date} because all values are NaN")
                        continue

                self.found.add(date)

            self.tried.update(to_try)

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
    def __init__(self, source, year, day):
        self.year = year
        self.day = day

    def transform(self, group_of_dates):
        from anemoi.datasets.dates.groups import GroupOfDates

        dates = list(group_of_dates)
        if not dates:
            return []

        new_dates = defaultdict(list)
        for date in dates:
            new_date = date.replace(year=self.year, day=self.day)
            new_dates[new_date].append(date)

        for date, dates in new_dates.items():
            yield (
                GroupOfDates([date], group_of_dates.provider),
                GroupOfDates(dates, group_of_dates.provider),
            )


class DateMapperConstant(DateMapper):
    def __init__(self, source, date=None):
        self.source = source
        self.date = date

    def transform(self, group_of_dates):
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
    def __init__(
        self,
        context,
        action_path,
        group_of_dates,
        source_result,
        mapper,
        original_group_of_dates,
    ):
        super().__init__(context, action_path, group_of_dates)

        self.source_results = source_result
        self.mapper = mapper
        self.original_group_of_dates = original_group_of_dates

    @property
    def datasource(self):
        result = []

        for field in self.source_results.datasource:
            for date in self.original_group_of_dates:
                result.append(NewValidDateTimeField(field, date))

        return FieldArray(result)


class RepeatedDatesAction(Action):
    def __init__(self, context, action_path, source, mode, **kwargs):
        super().__init__(context, action_path, source, mode, **kwargs)

        self.source = action_factory(source, context, action_path + ["source"])
        self.mapper = DateMapper.from_mode(mode, self.source, kwargs)

    @trace_select
    def select(self, group_of_dates):
        results = []
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

    def __repr__(self):
        return f"MultiDateMatchAction({self.source}, {self.mapper})"
