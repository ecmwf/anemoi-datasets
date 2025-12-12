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

from earthkit.data.utils.availability import Availability

from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate_utils.covering_intervals import covering_intervals

LOG = logging.getLogger(__name__)


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

    def candidate_intervals(self, current_time: datetime.datetime, **kwargs):
        # If not overridden in subclasses, the informations must be provided in the config
        raise NotImplementedError(f"Available periods must be provided in the config for {self.__class__.__name__}")

    def covering_intervals(self, start: datetime.datetime, end: datetime.datetime) -> list[SignedInterval]:
        candidates = self.hints or self.candidate_intervals

        intervals = covering_intervals(start, end, candidates, hints=self.hints)

        print(f"  Found covering intervals: for {start} to {end}:")
        for c in intervals:
            print(f"    {c}")
        return intervals


def match_fields_to_links(field, links: list[Link], debug=False):
    dprint = print if debug else lambda *args, **kwargs: None
    values = field.values

    date_str = str(field.metadata("validityDate")).zfill(8)
    time_str = str(field.metadata("validityTime")).zfill(4)
    valid_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")

    date_str = str(field.metadata("date")).zfill(8)
    time_str = str(field.metadata("time")).zfill(4)
    base_datetime = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    dprint(f"         field valid_date={valid_date}, base_datetime={base_datetime}")

    used = False
    for link in links:
        if valid_date != link.interval.max:
            dprint(f"  Skipping {link} as valid_date {valid_date} does not match {link.interval.max}")
            continue
        if link.interval.base is not None and base_datetime != link.interval.base:
            dprint(f"  Skipping {link} as base_datetime {base_datetime} does not match {link.interval.base}")
            continue
        if not link.accumulator.is_field_needed(field):
            dprint(f"  Skipping {link} as field not needed by its accumulator")
            continue
        dprint(f"   ✅ match found {field} sent to {link}")

        used = True
        yield (field, values, link)

    if not used:
        print(f"  ❌ field {field} not used by any accumulator")
        raise ValueError("unused field, stopping")
    else:
        dprint(f"   ✅ field {field} used by at least one accumulator")


class GribIndexCatalogue(Catalogue):
    def __init__(self, context, hints: dict, source: dict):
        super().__init__(context, hints, source)
        self.source_request = next(iter(source.values()))
        h = hashlib.md5(json.dumps(source, sort_keys=True).encode()).hexdigest()
        self.source_object = self.context.create_source(self.source, "data_sources", h)

    def retrieve_fields(self, links: list[Link], debug=False):
        dates = [link.interval.max for link in links]
        for field in self.source_object(self.context, dates):
            print("Processing field:", field)
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

    def retrieve_fields(self, links: list[Link], debug=False):
        dprint = print if debug else lambda *args, **kwargs: None

        requests = [self._build_request(link) for link in links]

        for r in requests:
            if self.context.use_grib_paramid and "param" in r:
                from anemoi.datasets.create.sources.mars import use_grib_paramid

                r = use_grib_paramid(r)

        dprint("💬 requests:")
        for req in requests:
            dprint("  request:", req)

        factorised = list(factorise_requests(requests))

        dprint("💬 factorised requests:")
        for req in factorised:
            dprint("  request:", req)

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


class EaOperCatalogue(MarsCatalogue):
    # todo : check wether these intervals are correct, or if it is 0-1/1-2/...
    candidate_intervals = [
        (6, "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12/12-13/13-14/14-15/15-16/16-17/17-18"),
        (18, "0-1/1-2/2-3/3-4/4-5/5-6/6-7/7-8/8-9/9-10/10-11/11-12/12-13/13-14/14-15/15-16/16-17/17-18"),
    ]


class EaEndaCatalogue(MarsCatalogue):
    candidate_intervals = [
        (6, "0-3/3-6/6-9/9-12/12-15/15-18"),
        (18, "0-3/3-6/6-9/9-12/12-15/15-18"),
    ]


class OdOperCatalogue(MarsCatalogue):
    # https://apps.ecmwf.int/mars-catalogue/?stream=oper&levtype=sfc&time=00%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-25&type=fc&class=od
    @property
    def candidate_intervals(self):
        steps = [(0, end) for end in list(range(1, 91)) + list(range(93, 145, 3)) + list(range(150, 241, 6))]
        return {0: steps, 12: steps}


class OdEldaCatalogue(MarsCatalogue):
    # https://apps.ecmwf.int/mars-catalogue/?stream=elda&levtype=sfc&time=06%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-31&type=fc&class=od
    @property
    def candidate_intervals(self):
        # something to do here related to anoffset?
        # ❌ not tested yet
        steps = [f"{0}-{i}" for i in range(1, 13)]
        return {6: steps, 18: steps}


class OdEnfoCatalogue(MarsCatalogue):
    # https://apps.ecmwf.int/mars-catalogue/?class=od&stream=enfo&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-31&time=06:00:00
    # not done. Use the config to provide available periods
    pass


class L5OperCatalogue(MarsCatalogue):
    # https://apps.ecmwf.int/mars-catalogue/?class=l5&stream=oper&expver=1&type=fc&year=2020&month=aug&levtype=sfc&date=2020-08-25&time=00:00:00
    @property
    def candidate_intervals(self):
        # ❌ not tested yet
        steps = [(0, i) for i in "/".split("1/2/3/4/5/6/9/12/15/18/21/24/27")]
        return [[base, steps] for base in [0, 3, 6, 9, 12, 15, 18, 21]]


class RrOperFrMsEcCatalogue(MarsCatalogue):
    # todo: check if the availability depends on the dates
    # https://apps.ecmwf.int/mars-catalogue/?origin=fr-ms-ec&stream=oper&levtype=sfc&time=06%3A00%3A00&expver=prod&month=aug&year=2020&date=2020-08-31&type=fc&class=rr
    # ❌ not tested yet
    candidate_intervals = [[0, [(0, i) for i in range(1, 22, 3)]]]


class RrOperSeAlEcCatalogue(MarsCatalogue):
    # https://apps.ecmwf.int/mars-catalogue/?class=rr&expver=prod&origin=se-al-ec&stream=oper&type=fc&year=2020&month=aug&levtype=sfc
    # ❌ not tested yet
    candidate_intervals = [[0, [(0, i) for i in (1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30)]]]


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

        class_ = source[source_name].get("class", None)
        stream = source[source_name].get("stream", None)
        origin = source[source_name].get("origin", None)

        match (class_, stream, origin):
            case ("ea", "oper", _) | ("ea", None, _):
                cls = EaOperCatalogue
            case ("ea", "enda", _):
                cls = EaEndaCatalogue
            case ("rr", "oper", "se-al-ec"):
                cls = RrOperSeAlEcCatalogue
            case ("rr", "oper", "fr-ms-ec"):
                cls = RrOperFrMsEcCatalogue
            # todo: implement these when needed
            # case ("rr", "oper", "no-ar-ce"):
            #     cls = RrOperNoArCeCatalogue
            # case ("rr", "oper", "no-ar-cw"):
            #     cls = RrOperNoArCwCatalogue
            # case ("rr", "oper", "no-ar-pa"):
            #     cls = RrOperNoArPaCatalogue
            case ("l5", "oper", _):
                cls = L5OperCatalogue
            case ("od", "oper", _):
                cls = OdOperCatalogue
            # case ("od", "enfo", _):
            #     cls = OdEnfoCatalogue
            case ("od", "elda", _):
                cls = OdEldaCatalogue
            case _:
                if not hints:
                    raise ValueError("More information are required for this type of requests")
                cls = MarsCatalogue

        return cls(context, hints, source)

    raise ValueError(f"Unknown source_name for catalogue: {source_name}")
