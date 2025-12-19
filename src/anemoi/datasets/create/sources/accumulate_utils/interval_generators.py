# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from abc import abstractmethod
from typing import Iterable

from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import covering_intervals

LOG = logging.getLogger(__name__)


def interval_without_base(current_time, start_step, end_step):
    start = datetime.datetime(current_time.year, current_time.month, current_time.day, start_step)
    end = start + datetime.timedelta(hours=end_step - start_step)
    return SignedInterval(start=start, end=end, base=None)


def interval_with_base(current_time, start_step, end_step, base_time):
    try:
        base_time = int(base_time)
    except ValueError:
        raise ValueError(f"Invalid base_time: {base_time} ({type(base_time)})")

    base = datetime.datetime(current_time.year, current_time.month, current_time.day, base_time)
    start = base + datetime.timedelta(hours=start_step)
    end = base + datetime.timedelta(hours=end_step)
    return SignedInterval(start=start, end=end, base=base)


class IntervalGenerator:
    @abstractmethod
    def covering_intervals(self, start: datetime.datetime, end: datetime.datetime) -> Iterable[SignedInterval]:
        pass


class Pattern:

    def __init__(self, base_time, steps, search_range=None, base_date=None):
        steps = normalise_steps(steps)
        self.steps = steps

        if base_time == "*":
            base_time = None
        self.base_time = base_time

        if search_range is None:
            search_range = [datetime.timedelta(days=d) for d in [-1, 0, 1]]
        else:
            search_range = [datetime.timedelta(days=d) for d in search_range]
        self.search_range = search_range

        if base_date:
            assert isinstance(base_date, dict), base_date
            assert "day_of_month" in base_date, base_date
            assert len(base_date) == 1, base_date
        self.base_date = base_date

    def filter(self, interval: SignedInterval):
        if self.base_date:
            if interval.base.day != self.base_date["day_of_month"]:
                return False
        return True


class SearchableIntervalGenerator(IntervalGenerator):
    def __init__(self, config):
        if isinstance(config, (tuple, list)):
            patterns = []
            for base_time, steps in config:
                patterns.append(Pattern(base_time=base_time, steps=steps))

        if isinstance(config, dict):
            patterns = [Pattern(**config)]

        self.patterns: list[Pattern] = patterns

    def covering_intervals(self, start: datetime.datetime, end: datetime.datetime) -> Iterable[SignedInterval]:
        return covering_intervals(start, end, self)

    def __call__(
        self,
        current_time: datetime.datetime,
        start: datetime.datetime,
        end: datetime.datetime,
        current_base: datetime.datetime,
    ) -> Iterable[SignedInterval]:
        # This generates starting or ending intervals for the given current_time
        del start, end, current_base

        intervals = []
        for p in self.patterns:
            search_range = p.search_range
            for delta in search_range:
                base_time = p.base_time
                steps = p.steps
                for start_step, end_step in steps:
                    if base_time is None:
                        interval = interval_without_base(current_time + delta, start_step, end_step)
                    else:
                        interval = interval_with_base(current_time + delta, start_step, end_step, base_time)
                        if not p.filter(interval):
                            continue

                    if interval not in intervals:
                        intervals.append(interval)

        # filter only the interval starting at current_time (or ending at current_time)
        filtered = []
        for i in intervals:
            if i.start == current_time:
                filtered.append(i)
            elif (-i).start == current_time:
                filtered.append(-i)
        intervals = filtered

        # quite important to sort by reversed base to prioritise most recent base in case of ties
        # in some cases, we may want to sort by other criteria
        intervals = sorted(intervals, key=lambda x: -(x.base or x.start).timestamp())

        return intervals


def normalise_steps(steps_list) -> list[list[int]]:
    res = []
    if isinstance(steps_list, str):
        steps_list = steps_list.split("/")
    assert isinstance(steps_list, list), steps_list

    for start_end_step in steps_list:
        if isinstance(start_end_step, str):
            assert "-" in start_end_step, start_end_step
            start_end_step = start_end_step.split("-")
        assert isinstance(start_end_step, (list, tuple)) and len(start_end_step) == 2, start_end_step
        start_step, end_step = int(start_end_step[0]), int(start_end_step[1])
        res.append([start_step, end_step])
    return res


class AccumulatedFromStartIntervalGenerator(SearchableIntervalGenerator):
    def __init__(self, basetime, frequency, last_step):
        config = []
        for base in basetime:
            for i in range(0, last_step, frequency):
                config.append([base, [f"0-{i+frequency}"]])
        super().__init__(config)


class AccumulatedFromPreviousStepIntervalGenerator(SearchableIntervalGenerator):
    def __init__(self, basetime, frequency, last_step):
        config = []
        for base in basetime:
            for i in range(0, last_step, frequency):
                config.append([base, [f"{i}-{i+frequency}"]])
        super().__init__(config)


def _interval_generator_factory(config) -> IntervalGenerator | list | dict:
    match config:
        case IntervalGenerator():
            return config

        case {"type": "accumulated-from-start", **params}:
            return AccumulatedFromStartIntervalGenerator(**params)
        case {"accumulated-from-start": params}:
            return AccumulatedFromStartIntervalGenerator(**params)

        case {"accumulated-from-previous-step": params}:
            return AccumulatedFromPreviousStepIntervalGenerator(**params)
        case {"type": "accumulated-from-previous-step", **params}:
            return AccumulatedFromPreviousStepIntervalGenerator(**params)

        case {"type": _, **params}:
            raise NotImplementedError(f"Unknown availability config {config}")

        case dict() | list() | tuple():
            return SearchableIntervalGenerator(config)

        case "era5-oper":
            return [
                (6, "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12/12-13/13-14/14-15/15-16/16-17/17-18"),
                (18, "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12/12-13/13-14/14-15/15-16/16-17/17-18"),
            ]
        case "era5-enda":
            return [
                (6, "0-3/3-6/6-9/9-12/12-15/15-18"),
                (18, "0-3/3-6/6-9/9-12/12-15/15-18"),
            ]

        case "od-oper":
            # https://apps.ecmwf.int/mars-catalogue/?stream=oper&levtype=sfc&time=00%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-25&type=fc&class=od
            steps = [f"{0}-{i}" for i in range(1, 91)]
            return ((0, steps), (12, steps))

        case "od-elda":
            # https://apps.ecmwf.int/mars-catalogue/?stream=elda&levtype=sfc&time=06%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-31&type=fc&class=od
            steps = [f"{0}-{i}" for i in range(1, 13)]
            return ((6, steps), (18, steps))

        case "od-enfo":
            # https://apps.ecmwf.int/mars-catalogue/?class=od&stream=enfo&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-31&time=06:00:00
            raise NotImplementedError("od-enfo interval generator not implemented yet")

        case "cerra-se-al-ec":
            # https://apps.ecmwf.int/mars-catalogue/?class=rr&expver=prod&origin=se-al-ec&stream=oper&type=fc&year=2020&month=aug&levtype=sfc
            return [[0, [(0, i) for i in [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30]]]]
        case "cerra-fr-ms-ec":
            # https://apps.ecmwf.int/mars-catalogue/?origin=fr-ms-ec&stream=oper&levtype=sfc&time=06%3A00%3A00&expver=prod&month=aug&year=2020&date=2020-08-31&type=fc&class=rr
            return [[0, [(0, i) for i in range(1, 22, 3)]]]

        case "l5-oper":
            # https://apps.ecmwf.int/mars-catalogue/?class=l5&stream=oper&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-25&time=00:00:00
            return [
                [base, [(0, i) for i in "/".split("1/2/3/4/5/6/9/12/15/18/21/24/27")]]
                for base in [0, 3, 6, 9, 12, 15, 18, 21]
            ]

        case str():
            try:
                data_accumulation_period = frequency_to_timedelta(config)
            except Exception as e:
                raise ValueError(f"Unknown interval generator config: {config}") from e

            hours = data_accumulation_period.total_seconds() / 3600
            if not (hours.is_integer() and hours > 0):
                raise ValueError("Only accumulation periods multiple of 1 hour are supported for now")

            return [["*", [f"{i}-{i+1}" for i in range(0, 24)]]]

        case _:
            raise ValueError(f"Unknown interval generator config: {config}")


def interval_generator_factory(config) -> IntervalGenerator:
    while not isinstance(config, IntervalGenerator):
        config = _interval_generator_factory(config)
    return config
