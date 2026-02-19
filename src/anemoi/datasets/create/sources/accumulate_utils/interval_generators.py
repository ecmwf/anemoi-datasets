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

from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import covering_intervals

LOG = logging.getLogger(__name__)


def build_interval(
    current_time: datetime.datetime, start_step: int, end_step: int, base_time: str | int | None
) -> SignedInterval:
    """Build a SignedInterval object corresponding to current_time's day
    This SignedInterval may not have a base datetime

    """
    try:
        usable_base_time = int(base_time) if base_time is not None else 0
    except ValueError:
        raise ValueError(f"Invalid base_time: {base_time} ({type(base_time)})")
    base = datetime.datetime(current_time.year, current_time.month, current_time.day, usable_base_time)
    start = base + datetime.timedelta(hours=start_step)
    end = base + datetime.timedelta(hours=end_step)

    interval_base = base if base_time is not None else None

    return SignedInterval(start=start, end=end, base=interval_base)


class IntervalGenerator:
    """Abstract base class to generate intervals.
    Call to IntervalGenerator will provide candidate intervals to be selected by the covering_intervals method
    """

    @abstractmethod
    def covering_intervals(self, start: datetime, end: datetime) -> Iterable[SignedInterval]:
        pass

    @abstractmethod
    def __call__():
        pass


class Pattern:
    """Common format of config arguments to build SearchableIntervalGenerator
    Used to supply candidate intervals for IntervalGenerator
    """

    def __init__(
        self,
        base_time: str | datetime.datetime,
        steps: str | list[str],
        search_range: list[int] | None = None,
        base_date: dict | None = None,
    ):
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

    def filter(self, interval: SignedInterval) -> bool:
        if self.base_date:
            if interval.base.day != self.base_date["day_of_month"]:
                return False
        return True


class SearchableIntervalGenerator(IntervalGenerator):
    def __init__(self, config: tuple | list | dict):
        if isinstance(config, (tuple, list)):
            patterns = []
            for base_time, steps in config:
                patterns.append(Pattern(base_time=base_time, steps=steps))

        if isinstance(config, dict):
            patterns = [Pattern(**config)]

        self.patterns: list[Pattern] = patterns

    def covering_intervals(self, start: datetime.datetime, end: datetime.datetime) -> Iterable[SignedInterval]:
        """Perform interval search among candidates with minimal base switches and length.
        Candidates are given by self call
        Return available SignedIntervals covering the period start->end (where start>end is possible)
        """
        return covering_intervals(start, end, self)

    def __call__(
        self,
        current_time: datetime.datetime,
    ) -> Iterable[SignedInterval]:
        """This generates candidate intervals starting or ending at the given current_time
        Candidates correspond to the pairs of base_time, steps stored in patterns
        """
        intervals = []
        for p in self.patterns:
            search_range = p.search_range
            for delta in search_range:
                base_time = p.base_time
                steps = p.steps
                for start_step, end_step in steps:
                    interval = build_interval(current_time + delta, start_step, end_step, base_time)
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


class CycleIntervalProvider(SearchableIntervalGenerator):
    def __init__(self, **config: dict):
        print(f"CycleIntervalProvider config: {config}")
        self.reference = config.pop("start", datetime.datetime(1970, 1, 1, 0, 0))
        self.reference = as_datetime(self.reference)

        def split(s):
            i, j = s.split("-")
            return int(i), int(j)

        def normalise_steps(base_time, steps):
            steps = steps.split("/")
            return base_time, [split(s) for s in steps]

        self.config = {split(k): normalise_steps(*v) for k, v in config.items()}

    def covering_intervals(self, start: datetime.datetime, end: datetime.datetime) -> Iterable[SignedInterval]:
        cycle_length_in_hours = max([k[1] for k in self.config.keys()])

        i_start = (int((start - self.reference).total_seconds()) // 3600) % cycle_length_in_hours
        i_end = (int((end - self.reference).total_seconds()) // 3600) % cycle_length_in_hours
        if i_end == 0:
            i_end = cycle_length_in_hours

        if (i_start, i_end) not in self.config:
            raise ValueError(
                f"CycleIntervalProvider: no config to find ({i_start}, {i_end}) (start={start}, end={end}, {cycle_length_in_hours=})"
            )

        base_time, steps = self.config[(i_start, i_end)]

        base_datetime = datetime.datetime(end.year, end.month, end.day, base_time)
        # The base must be strictly before end so step 0-N lands on or before end.
        while base_datetime >= end:
            base_datetime -= datetime.timedelta(days=1)

        assert (
            base_datetime.hour == base_time
        ), f"Base datetime hour {base_datetime.hour} does not match expected base time {base_time}"

        intervals = []
        for start_step, end_step in steps:
            interval = SignedInterval(
                base=base_datetime,
                start=base_datetime + datetime.timedelta(hours=start_step),
                end=base_datetime + datetime.timedelta(hours=end_step),
            )
            intervals.append(interval)

        if not (any(i.start == start for i in intervals) or any((-i).start == start for i in intervals)):
            raise ValueError(
                f"CycleIntervalProvider: no interval starting at {start} (start={start}, end={end}, {cycle_length_in_hours=})"
            )
        if not (any(i.end == end for i in intervals) or any((-i).end == end for i in intervals)):
            raise ValueError(
                f"CycleIntervalProvider: no interval ending at {end} (start={start}, end={end}, {cycle_length_in_hours=})"
            )

        return intervals


def normalise_steps(steps_list: str | list[str]) -> list[list[int]]:
    """Convert the input step_list to a list of [start,end] pairs"""
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
    def __init__(self, basetime: str | datetime.datetime, frequency: int, last_step: int):
        config = []
        for base in basetime:
            for i in range(0, last_step, frequency):
                config.append([base, [f"0-{i+frequency}"]])
        super().__init__(config)


class AccumulatedFromPreviousStepIntervalGenerator(SearchableIntervalGenerator):
    def __init__(self, basetime: str | datetime.datetime, frequency: int, last_step: int):
        config = []
        for base in basetime:
            for i in range(0, last_step, frequency):
                config.append([base, [f"{i}-{i+frequency}"]])
        super().__init__(config)


def _match_mars_config(_class: str, _stream: str | None = None, _origin: str | None = None) -> list | tuple:
    """Match MARS configuration (class, stream, origin) to interval generator config.

    Parameters
    ----------
    _class
        MARS class (e.g., 'ea', 'od', 'rr', 'l5').
    _stream
        MARS stream (e.g., 'oper', 'enda', 'elda', 'enfo'). Defaults to 'oper'.
    _origin
        MARS origin (e.g., 'se-al-ec', 'fr-ms-ec'). Defaults to None.

    Returns
    -------
    list | tuple
        Interval generator configuration.

    Raises
    ------
    NotImplementedError
        If the combination is not yet implemented.
    ValueError
        If the combination is unknown.
    """
    _stream = _stream or "oper"

    match (_class, _stream, _origin):
        case ("ea", "oper", _):
            return [
                (6, "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12/12-13/13-14/14-15/15-16/16-17/17-18"),
                (18, "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12/12-13/13-14/14-15/15-16/16-17/17-18"),
            ]
        case ("ea", "enda", _):
            return [
                (6, "0-3/3-6/6-9/9-12/12-15/15-18"),
                (18, "0-3/3-6/6-9/9-12/12-15/15-18"),
            ]

        case ("od", "oper", _):
            # https://apps.ecmwf.int/mars-catalogue/?stream=oper&levtype=sfc&time=00%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-25&type=fc&class=od
            steps = [f"{0}-{i}" for i in range(1, 91)]
            return ((0, steps), (12, steps))
        case ("od", "elda", _):
            # https://apps.ecmwf.int/mars-catalogue/?stream=elda&levtype=sfc&time=06%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-31&type=fc&class=od
            # (6,  "0-1/0-2/0-3/0-4/0-5/0-6/0-7/0-8/0-9/0-10/0-11/0-12"),
            # (18, "0-1/0-2/0-3/0-4/0-5/0-6/0-7/0-8/0-9/0-10/0-11/0-12")
            steps = [f"{0}-{i}" for i in range(1, 13)]
            return ((6, steps), (18, steps))
        case ("od", "enfo", _):
            # https://apps.ecmwf.int/mars-catalogue/?class=od&stream=enfo&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-31&time=06:00:00
            raise NotImplementedError("od-enfo interval generator not implemented yet")

        case ("rr", _, "se-al-ec"):
            # https://apps.ecmwf.int/mars-catalogue/?class=rr&expver=prod&origin=se-al-ec&stream=oper&type=fc&year=2020&month=aug&levtype=sfc
            return [[0, [(0, i) for i in [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30]]]]
        case ("rr", _, "fr-ms-ec"):
            # https://apps.ecmwf.int/mars-catalogue/?origin=fr-ms-ec&stream=oper&levtype=sfc&time=06%3A00%3A00&expver=prod&month=aug&year=2020&date=2020-08-31&type=fc&class=rr
            return [[0, [(0, i) for i in range(1, 22, 3)]]]

        case ("l5", "oper", _):
            # https://apps.ecmwf.int/mars-catalogue/?class=l5&stream=oper&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-25&time=00:00:00
            return [[0, [(int(i), int(i) + 1) for i in range(0, 24)]]]

        case _:
            raise ValueError(f"Unknown MARS configuration: class={_class}, stream={_stream}, origin={_origin}")


def _interval_generator_factory(
    config, source_name: str | None = None, source: dict | None = None
) -> IntervalGenerator | list | dict:
    match config:
        case IntervalGenerator():
            return config

        case {"type": "accumulated-from-start", **params}:
            return AccumulatedFromStartIntervalGenerator(**params)
        case {"accumulated-from-start": params}:
            return AccumulatedFromStartIntervalGenerator(**params)

        case {"type": "cycle", **params}:
            return CycleIntervalProvider(**params)
        case {"cycle": params}:
            return CycleIntervalProvider(**params)

        case {"accumulated-from-previous-step": params}:
            return AccumulatedFromPreviousStepIntervalGenerator(**params)
        case {"type": "accumulated-from-previous-step", **params}:
            return AccumulatedFromPreviousStepIntervalGenerator(**params)

        case {"type": _, **params}:
            raise NotImplementedError(f"Unknown availability config {config}")

        case {"mars": mars_config}:
            _class = mars_config.get("class")
            _stream = mars_config.get("stream")
            _origin = mars_config.get("origin")
            assert _class is not None, "mars config must have a 'class' key"
            return _match_mars_config(_class, _stream, _origin)

        case dict() | list() | tuple():
            return SearchableIntervalGenerator(config)

        case "auto":
            assert None not in (source_name, source), "Source must be specified when using 'auto' discovery"
            assert source_name == "mars", "Only 'mars' source is currently supported for 'auto' availability discovery"

            _class, _stream, _origin = source.get("class", None), source.get("stream", None), source.get("origin", None)

            assert (
                _class is not None
            ), "Availability should be automatically determined from mars source, but the mars source has no 'class'"

            if (_stream is None) or (_origin is None):
                LOG.warning(
                    f"Stream and/or origin unspecified for class {_class}, "
                    f"stream and/or origin will be set as defaults.",
                )

            return {"mars": {"class": _class, "stream": _stream, "origin": _origin}}

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


def interval_generator_factory(config, source_name: str | None = None, source: dict | None = None) -> IntervalGenerator:
    while not isinstance(config, IntervalGenerator):
        config = _interval_generator_factory(config, source_name, source)
    return config
