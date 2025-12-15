# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import hashlib
import json
import logging
from copy import deepcopy
from typing import Any
from typing import Iterable
from typing import Optional

from earthkit.data.utils.availability import Availability

from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import covering_intervals

LOG = logging.getLogger(__name__)
DEBUG = False  # True
trace = print if DEBUG else lambda *args, **kwargs: None


def _member(field: Any) -> int:
    """Retrieves the member number from the field metadata.

    Parameters:
    ----------
    field : Any
        The field from which to retrieve the member number.

    Return:
    -------
    int
        The member number.
    """
    # Bug in eccodes has number=0 randomly
    number = field.metadata("number", default=0)
    if number is None:
        number = 0
    return number


def _to_list(x: list[Any] | tuple[Any] | Any) -> list[Any]:
    """Converts the input to a list if it is not already a list or tuple.

    Parameters:
    ----------
    x : Union[List[Any], Tuple[Any], Any]
        Input value.

    Return:
    -------
    List[Any]
        The input value as a list.
    """
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def _scda(request: dict[str, Any]) -> dict[str, Any]:
    """Modifies the request stream based on the time.

    Parameters:
    ----------
    request : dict[str, Any]
        Request parameters.

    Return:
    -------
    dict[str, Any]
        The modified request parameters.
    """
    if request["time"] in (6, 18, 600, 1800):
        request["stream"] = "scda"
    else:
        request["stream"] = "oper"
    return request


def factorise_requests(requests):
    compressed = Availability(requests)
    for r in compressed.iterate():
        for k, v in r.items():
            if isinstance(v, (list, tuple)) and len(v) == 1:
                r[k] = v[0]
        yield r


class Link:
    def __init__(self, interval, accumulator, catalogue):
        self.catalogue = catalogue
        self.accumulator = accumulator
        self.interval = interval

    def __repr__(self):
        return f"Link({self.interval})"


class Catalogue:
    def __init__(self, context, hints: dict, source: dict):
        self.context = context
        self.hints = hints
        self.source = source

    def covering_intervals(self, start: datetime.datetime, end: datetime.datetime) -> list[SignedInterval]:
        intervals = covering_intervals(start, end, self.hints, hints=self.hints)

        LOG.debug(f"  Found covering intervals: for {start} to {end}:")
        for c in intervals:
            LOG.debug(f"    {c}")
        return intervals


def match_fields_to_links(field, links: list[Link]):
    values = field.values

    date_str = str(field.metadata("validityDate")).zfill(8)
    time_str = str(field.metadata("validityTime")).zfill(4)
    valid_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")

    date_str = str(field.metadata("date")).zfill(8)
    time_str = str(field.metadata("time")).zfill(4)
    base_datetime = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    trace("  Matching field:", field)

    endStep = field.metadata("endStep")
    startStep = field.metadata("startStep")
    typeStep = field.metadata("stepType")

    # if endStep == 0 and startStep > 0:
    #   startStep, endStep = endStep, startStep

    if startStep == endStep:
        startStep = 0
        assert typeStep == "instant", "If startStep == endStep, stepType must be 'instant'"
    assert startStep < endStep, (startStep, endStep)

    length = datetime.timedelta(hours=endStep) - datetime.timedelta(hours=startStep)

    assert valid_date == base_datetime + datetime.timedelta(hours=endStep)

    print(
        f"    field: valid_date={valid_date}, base_datetime={base_datetime}, startStep={startStep}, endStep={endStep}, length={length}"
    )

    used = False
    for link in links:

        if valid_date != link.interval.max:
            trace(f"  Skipping {link} as valid_date {valid_date} does not match {link.interval.max}")
            continue

        if length != (link.interval.max - link.interval.min):
            trace(f"  Skipping {link} as length {length} does not match {link.interval}")
            continue

        if link.interval.base is not None:
            # forecast interval, basetime is defined
            if base_datetime != link.interval.base:
                trace(f"  Skipping {link} as base_datetime {base_datetime} does not match {link.interval.base}")
                continue
            assert endStep == (link.interval.max - link.interval.base).total_seconds() / 3600, (endStep, link.interval)
            assert startStep == (link.interval.min - link.interval.base).total_seconds() / 3600, (
                startStep,
                link.interval,
            )

        if not link.accumulator.is_field_needed(field):
            trace(f"  Skipping {link} as field not needed by its accumulator")
            continue

        # some extra checks for paranoia
        assert valid_date == link.interval.max, (valid_date, link.interval)
        assert length == (link.interval.max - link.interval.min), (length, link.interval)
        if link.interval.base is not None:
            assert base_datetime == link.interval.base, (base_datetime, link.interval)

        trace(f"   ✅ match found {field} sent to {link}")

        used = True
        yield (field, values, link)

    if not used:
        LOG.error(f"  ❌ field {field} not used by any accumulator")
        raise ValueError(f"unused field {field}, stopping")
    else:
        trace(f"   ✅ field {field} used by at least one accumulator")


class GribIndexCatalogue(Catalogue):
    def __init__(self, context, hints: dict, source: dict):
        super().__init__(context, hints, source)
        self.source_request = next(iter(source.values()))
        h = hashlib.md5(json.dumps(source, sort_keys=True).encode()).hexdigest()
        self.source_object = self.context.create_source(self.source, "data_sources", h)

    def retrieve_fields(self, links: list[Link], debug=False):
        dates = [link.interval.max for link in links]
        for field in self.source_object(self.context, dates):
            LOG.debug("Processing field:", field)
            yield from match_fields_to_links(field, links)

    def get_all_keys(self):
        main_request, param, number, additional = _normalise_request(self.source_request)
        for p in param:
            #    for n in number:
            yield dict(param=p)

    # use this ?
    def search_possible_intervals(self, start, end, debug=False):
        fields = self.source_object(self.context, [start, end])
        intervals = []
        for field in fields:
            if field.metadata("stepType") != "accum":
                continue
            date = str(field.metadata("date")).zfill(8)
            time = str(field.metadata("time")).zfill(4)
            base_datetime = datetime.datetime.strptime(f"{date}{time}", "%Y%m%d%H%M")

            start_step, end_step = field.metadata("startStep"), field.metadata("endStep")
            start_step, end_step = datetime.timedelta(hours=start_step), datetime.timedelta(hours=end_step)
            intervals.append(SignedInterval(base_datetime + start_step, base_datetime + end_step, base_datetime))
        return intervals


def _normalise_request(request: dict[str, Any]) -> dict[str, Any]:
    request = deepcopy(request)

    param = request.pop("param")
    assert isinstance(param, (list, tuple))

    number = request.pop("number", [0])
    if not isinstance(number, (list, tuple)):
        number = [number]
    assert isinstance(number, (list, tuple))

    stream = request.pop("stream", "oper")

    type_ = request.pop("type", "an")
    if type_ == "an":
        type_ = "fc"

    levtype = request.pop("levtype", "sfc")
    if levtype != "sfc":
        raise NotImplementedError("Only sfc leveltype is supported")

    additional_request = {"stream": stream, "type": type_, "levtype": levtype}

    return request, param, number, additional_request


class MarsCatalogue(Catalogue):
    def __init__(self, context, hints: dict, source: dict):
        super().__init__(context, hints, source)
        _, self.source_request = next(iter(source.items()))

    def request_to_fields(self, req: dict) -> list:
        from earthkit.data import from_source

        return from_source("mars", req)

    def _build_request(self, link) -> dict:
        interval = link.interval
        assert isinstance(interval, SignedInterval), type(interval)
        request = {}
        for k, v in self.source_request.items():
            request[k] = v
        for k, v in link.accumulator.key.items():
            request[k] = v
        step = interval.max - interval.base
        hours = step.total_seconds() // 3600
        minutes = (step.total_seconds() // 60) % 60
        assert minutes == 0, "Only full hours supported in grib/mars"
        request["step"] = int(hours)
        request["date"] = interval.base.strftime("%Y%m%d")
        request["time"] = interval.base.strftime("%H%M")
        return request

    def retrieve_fields(self, links: list[Link], debug=False):
        requests = [self._build_request(link) for link in links]

        for r in requests:
            if self.context.use_grib_paramid and "param" in r:
                from anemoi.datasets.create.sources.mars import use_grib_paramid

                r = use_grib_paramid(r)

        trace("💬 requests:")
        for req in requests:
            trace("  request:", req)

        factorised = list(factorise_requests(requests))

        trace("💬 factorised requests:")
        for req in factorised:
            trace("  request:", req)

        ds = self.context.empty_result()
        for req in factorised:
            ds += self.request_to_fields(req)

        for field in ds:
            yield from match_fields_to_links(field, links)

    def get_all_keys(self):
        main_request, param, number, additional = _normalise_request(self.source_request)
        for p in param:
            for n in number:
                # add level here if needed
                yield dict(param=p, number=n)


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
        hints: Optional[datetime.datetime],
    ) -> Iterable[SignedInterval]:
        # Using the config list provided, this generates starting or ending intervals
        # for the given current_time
        # it follows the API defined in covering_intervals
        #
        # support for non-hourly steps could be added later if needed
        del hints
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


def interval_generator_factory(config) -> IntervalGenerator:
    while not isinstance(config, IntervalGenerator):
        config = _interval_generator_factory(config)
    return config


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
        case _:
            raise ValueError(f"Unknown interval generator config: {config}")


def build_catalogue(context, hints: dict, source) -> Catalogue:
    assert isinstance(source, dict)
    assert len(source) == 1, f"Source must have exactly one key, got {list(source.keys())}"

    source_name, _ = next(iter(source.items()))

    if source_name == "grib-index":
        return GribIndexCatalogue(context, hints, source)

    if source_name == "mars":
        if "type" not in source[source_name]:
            source[source_name]["type"] = "fc"
            LOG.warning("Assuming 'type: fc' for mars source as it was not specified in the recipe")

        if "levtype" not in source[source_name]:
            source[source_name]["levtype"] = "sfc"
            LOG.warning("Assuming 'levtype: sfc' for mars source as it was not specified in the recipe")

        interval_generator = interval_generator_factory(hints)
        return MarsCatalogue(context, interval_generator, source)

    raise ValueError(f"Unknown source_name for catalogue: {source_name}")
