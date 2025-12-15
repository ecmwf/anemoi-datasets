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
from typing import Iterable

from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import SignedInterval

LOG = logging.getLogger(__name__)


class IntervalGenerator:
    def __init__(self, func):
        if not callable(func):
            func = _normalise_candidates_function(func)
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _normalise_candidates_function(config):
    assert isinstance(config, list), (type(config), config)

    def interval_without_base(current_time, delta, steps):
        start = datetime.datetime(current_time.year, current_time.month, current_time.day, steps[0]) + delta
        end = start + datetime.timedelta(hours=steps[1] - steps[0])
        return SignedInterval(start=start, end=end, base=None)

    def interval_with_base(current_time, delta, steps, base_hour):
        try:
            base_hour = int(base_hour)
        except ValueError:
            raise ValueError(f"Invalid base_hour: {base_hour} ({type(base_hour)})")

        base = datetime.datetime(current_time.year, current_time.month, current_time.day, base_hour) + delta
        start = base + datetime.timedelta(hours=steps[0])
        end = base + datetime.timedelta(hours=steps[1])
        return SignedInterval(start=start, end=end, base=base)

    def candidates(
        current_time: datetime.datetime,
        start: datetime.datetime,
        end: datetime.datetime,
        current_base: datetime.datetime,
    ) -> Iterable[SignedInterval]:
        # Using the config list provided, this generates starting or ending intervals
        # for the given current_time
        # it follows the API defined in covering_intervals
        #
        # support for non-hourly steps could be added later if needed
        del start
        del end
        del current_base

        # we could have "extend_to_deltas" in config, but for now we just hardcode
        # if we do that, we need to find a better name than "extend_to_deltas"
        extend_to_deltas = [datetime.timedelta(days=d) for d in [-1, 0, 1]]

        if not isinstance(config, (tuple, list)):
            raise ValueError(f"Expected config to be a list or tuple, got {type(config)}: {config}")
        for _ in config:
            if not isinstance(_, (list, tuple)):
                raise ValueError(f"Invalid config entry: {_} has type({type(_)}) in {config=}")
            if len(_) != 2:
                raise ValueError(f"Invalid config entry: {_} has length {len(_)} in {config=}")

        intervals = []
        for delta in extend_to_deltas:
            for base_hour, steps_list in config:
                if isinstance(steps_list, str):
                    steps_list = steps_list.split("/")
                assert isinstance(steps_list, list), steps_list
                for steps in steps_list:
                    if isinstance(steps, str):
                        assert "-" in steps, steps
                        steps = tuple(map(int, steps.split("-")))
                    assert isinstance(steps, tuple) and len(steps) == 2, steps

                    if base_hour == "*":
                        base_hour = None

                    if base_hour is None:
                        intervals.append(interval_without_base(current_time, delta, steps))
                        continue
                    intervals.append(interval_with_base(current_time, delta, steps, base_hour))

        intervals = [i for i in intervals if i.start == current_time or current_time == i.end]

        # quite important to sort by -base.timestamp() to prioritise most recent base in case of ties
        # in some cases, we may want to sort by other criteria
        intervals = sorted(intervals, key=lambda x: -(x.base or x.start).timestamp())

        return intervals

    return candidates


def _interval_generator_factory(config) -> IntervalGenerator | list | dict:
    match config:
        case IntervalGenerator():
            return config

        case dict():
            type_ = config.get("type", None)
            raise NotImplementedError(f"IntervalGenerator of type {type_} is not implemented yet")

        case list() | tuple():
            return IntervalGenerator(config)

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
            return [(0, end) for end in list(range(1, 91))]

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
