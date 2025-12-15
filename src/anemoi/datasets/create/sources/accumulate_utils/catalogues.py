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
from typing import TYPE_CHECKING
from typing import Any

from earthkit.data.utils.availability import Availability

if TYPE_CHECKING:
    from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import SignedInterval

from anemoi.datasets.create.sources.accumulate_utils.interval_generators import interval_generator_factory

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
    def __init__(self, context, interval_generator: dict, source: dict):
        self.context = context
        self.interval_generator = interval_generator
        self.source = source

    def covering_intervals(self, start: datetime.datetime, end: datetime.datetime) -> list["SignedInterval"]:
        intervals = self.interval_generator.covering_intervals(start, end)

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
    def __init__(self, context, interval_generator: dict, source: dict):
        super().__init__(context, interval_generator, source)
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
    # def search_possible_intervals(self, start, end, debug=False):
    #     fields = self.source_object(self.context, [start, end])
    #     intervals = []
    #     for field in fields:
    #         if field.metadata("stepType") != "accum":
    #             continue
    #         date = str(field.metadata("date")).zfill(8)
    #         time = str(field.metadata("time")).zfill(4)
    #         base_datetime = datetime.datetime.strptime(f"{date}{time}", "%Y%m%d%H%M")

    #         start_step, end_step = field.metadata("startStep"), field.metadata("endStep")
    #         start_step, end_step = datetime.timedelta(hours=start_step), datetime.timedelta(hours=end_step)
    #         intervals.append(SignedInterval(base_datetime + start_step, base_datetime + end_step, base_datetime))
    #     return intervals


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
    def __init__(self, context, interval_generator: dict, source: dict):
        super().__init__(context, interval_generator, source)
        _, self.source_request = next(iter(source.items()))

    def request_to_fields(self, req: dict) -> list:
        from earthkit.data import from_source

        return from_source("mars", req)

    def _build_request(self, link) -> dict:
        interval = link.interval
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


def build_catalogue(context, interval_generator: dict, source) -> Catalogue:
    assert isinstance(source, dict)
    assert len(source) == 1, f"Source must have exactly one key, got {list(source.keys())}"

    source_name, _ = next(iter(source.items()))
    interval_generator = interval_generator_factory(interval_generator)

    if source_name == "grib-index":
        return GribIndexCatalogue(context, interval_generator, source)

    if source_name == "mars":
        if "type" not in source[source_name]:
            source[source_name]["type"] = "fc"
            LOG.warning("Assuming 'type: fc' for mars source as it was not specified in the recipe")

        if "levtype" not in source[source_name]:
            source[source_name]["levtype"] = "sfc"
            LOG.warning("Assuming 'levtype: sfc' for mars source as it was not specified in the recipe")

        return MarsCatalogue(context, interval_generator, source)

    raise ValueError(f"Unknown source_name for catalogue: {source_name}")
