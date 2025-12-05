# (C) Copyright 2024 Anemoi contributors.
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

import numpy as np
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.utils.availability import Availability

from anemoi.datasets.create.sources.covering_intervals import covering_intervals

LOG = logging.getLogger(__name__)


def factorise_requests(requests):
    compressed = Availability(requests)
    for r in compressed.iterate():
        for k, v in r.items():
            if isinstance(v, (list, tuple)) and len(v) == 1:
                r[k] = v[0]
        yield r


class Vector:
    def __init__(self, start, end):
        self.is_null = start == end

        self.start = start
        self.end = end
        if start < end:
            self.min = start
            self.max = end
            self.sign = 1
        else:
            self.min = end
            self.max = start
            self.sign = -1

    @classmethod
    def null_vector(cls):
        return cls(datetime.datetime(1970, 1, 1), datetime.datetime(1970, 1, 1))

    @classmethod
    def from_valid_date_and_period(cls, valid_date: datetime.datetime, period: datetime.timedelta | str):
        if not isinstance(valid_date, datetime.datetime):
            raise TypeError("valid_date must be a datetime.datetime instance")
        if not isinstance(period, datetime.timedelta):
            period = frequency_to_timedelta(period)
        start = valid_date - period
        end = valid_date
        return Vector(start, end)

    def __eq__(self, other: Any):
        if not isinstance(other, Vector):
            return False
        return (self.start == other.start) and (self.end == other.end)

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self, *args):
        extra = ", " + ", ".join(str(a) for a in args) if args else ""
        GREEN = "\033[92m"
        BLUE = "\033[94m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"
        start = f"{BLUE}{self.start.strftime('%Y%m%d.%H%M')}{RESET}"

        if self.start < self.end:
            period = f"{GREEN}+{frequency_to_string(self.end - self.start)}{RESET}"
        elif self.start == self.end:
            period = f"{YELLOW}0s{RESET}"
        else:
            period = f"{RED}-{frequency_to_string(self.start - self.end)}{RESET}"

        end = f"{BLUE}{self.end.strftime('%Y%m%d.%H%M')}{RESET}"
        return f"{self.__class__.__name__}({start}{period}->{end}{extra})"


class ForecastVector(Vector):

    def __init__(self, start, end, base: datetime.datetime):
        """Defining an insecable time segment carrying all temporal information.

        base: datetime.datetime
            the date and hour referring to the beginning of the forecast run that has produced the data
            At this base, the accumulation of the forecast is zero.
            In the case of DefaultIntervalsCollections, this is equal to start_datetime.
        """
        super().__init__(start, end)
        self.base = base

        assert base <= self.start, f"invalid base date for ForecastVector: {base} > {self.start}"
        assert base <= self.end, f"invalid base date for ForecastVector: {base} > {self.end}"

    def __eq__(self, other: Any):
        if not isinstance(other, ForecastVector):
            return False
        if not self.base != other.base:
            return False
        return super().__eq__(other)

    def __hash__(self):
        return hash((super().__hash__(), self.base))

    def __repr__(self):
        step1 = frequency_to_string(self.min - self.base)
        step2 = frequency_to_string(self.max - self.base)
        return super().__repr__(f"basetime={self.base}, steps=[{step1}->{step2}]")


class VectorCollection(set):
    @property
    def granularity(self):
        return datetime.timedelta(seconds=3600)

    def sum(self, debug=False) -> Vector:
        """Compute the sum of all vectors in the collection.

        Returns:
        -------
        Vector
            A new Vector representing the sum of all vectors in the collection.
        """
        if not self:
            LOG.warning("Computing sum of an empty VectorCollection, returning null vector")
            return Vector.null_vector()

        minimum = min(v.start for v in self)
        maximum = max(v.end for v in self)

        if debug:
            print("Computing sum of VectorCollection:")
            for v in self:
                print(f"  {v}")

        for v in self:
            for d in [v.start, v.end]:
                delta = d - minimum
                if delta.total_seconds() % self.granularity.total_seconds() != 0:
                    raise NotImplementedError(f"{delta} is not a multiple of {self.granularity}, for vector {v}")

        def to_i(dt: datetime.datetime) -> int:
            seconds = (dt - minimum).total_seconds()
            i = int(seconds // self.granularity.total_seconds())
            assert seconds == i * self.granularity.total_seconds(), (dt, minimum, self.granularity, seconds, i)
            return i

        assert to_i(minimum) == 0, to_i(minimum)

        mini = to_i(minimum)
        maxi = to_i(maximum)
        x = np.zeros(maxi - mini, dtype=int)
        if debug:
            print("Coverage array:", x)
        for v in self:
            start_i = to_i(v.min)
            end_i = to_i(v.max)
            x[start_i:end_i] += v.sign
            if debug:
                print(f"  Applying vector {v}, start_i={start_i}, end_i={end_i}, sign={v.sign}")
                print("Coverage array:", x)

        # if x is [0, 0, 0, 1, 1, 1, 0, 0, 0] for example, return the covering vector corresponding to [3, 6]
        # if x has gaps, cannot compute a single covering vector, then return self
        x_nonzero = np.nonzero(x)[0]
        if len(x_nonzero) == 0:
            LOG.warning("The sum of vectors results in an empty coverage, returning null vector")
            return Vector.null_vector()
        first_nonzero = x_nonzero[0]
        last_nonzero = x_nonzero[-1] + 1
        if not np.all(x[first_nonzero:last_nonzero] == 1):
            LOG.warning("The sum of vectors results in gaps, cannot compute a single covering vector")
            # we could simplify the collection summing some of the vectors
            # but this is not necessary for now
            # this could be implemented later if needed, for optimization to avoid recomputing
            # this sum every time
            return self

        start = minimum + datetime.timedelta(seconds=first_nonzero * self.granularity.total_seconds())
        end = minimum + datetime.timedelta(seconds=last_nonzero * self.granularity.total_seconds())
        return Vector(start, end)


class RrOperIntervalsCollection:
    forecast_period: datetime.timedelta = datetime.timedelta(hours=3)

    def available_steps(self) -> dict:
        x = 24  # todo: check if 24 is the right value
        return {
            0: [[0, i] for i in range(0, x, 1)],
            3: [[0, i] for i in range(0, x, 1)],
            6: [[0, i] for i in range(0, x, 1)],
            9: [[0, i] for i in range(0, x, 1)],
            12: [[0, i] for i in range(0, x, 1)],
            15: [[0, i] for i in range(0, x, 1)],
            18: [[0, i] for i in range(0, x, 1)],
            21: [[0, i] for i in range(0, x, 1)],
        }


class OdEldaIntervalsCollection:
    def available_steps(self, start, end) -> dict:
        x = 24  # need to check if 24 is the right value
        return {
            6: [[i, i + 1] for i in range(0, x, 1)],
            18: [[i, i + 1] for i in range(0, x, 1)],
        }


class Link:
    def __init__(self, vector, accumulator, catalogue):
        self.catalogue = catalogue
        self.accumulator = accumulator
        self.vector = vector

    def __repr__(self):
        return f"Link({self.vector})"

    def __hash__(self):
        return hash((self.vector, self.catalogue, self.accumulator))


class LinksCollection(list):

    def match_field(self, field: Any) -> list[Link]:
        date_str = str(field.metadata("validityDate")).zfill(8)
        time_str = str(field.metadata("validityTime")).zfill(4)
        valid_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        print(f"Field valid_date: {valid_date}")
        print("because")
        print(f"  validityDate: {field.metadata('validityDate')}, validityTime: {field.metadata('validityTime')}")
        print("and")
        print(f"  date_str: {date_str}, time_str: {time_str}")

        assert field.metadata("stepType") == "accum", f"Not an accumulated variable {field.metadata('stepType')}"
        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")
        assert isinstance(startStep, int)
        assert isinstance(endStep, int)
        start_datetime = valid_date - datetime.timedelta(hours=endStep)
        end_datetime = valid_date - datetime.timedelta(hours=startStep)
        print("AND:")
        print(f"  valid_date: {valid_date}")
        print(f"  startStep: {startStep}, endStep: {endStep}")
        print(f"  which gives start_datetime: {start_datetime}, end_datetime: {end_datetime}")

        print("🔍🔍🔍")
        print("Searching Link matching this field:", field)
        print(f"  field valid_date={valid_date}, start_datetime={start_datetime}, end_datetime={end_datetime}")
        found = []
        for link in self:
            assert isinstance(link, Link)
            if not (link.valid_date == valid_date):
                print(f"  Skipping {link} as valid_date does not match {valid_date}")
                continue
            if not (link.interval.start_datetime == start_datetime):
                print(f"  Skipping {link} as start_datetime does not match {start_datetime}")
                continue
            if not (link.interval.end_datetime == end_datetime):
                print(f"  Skipping {link} as end_datetime does not match {end_datetime}")
                continue
            found.append(link)
        return found

    def select(self, filter):
        """Select links matching the given key filter."""
        return LinksCollection(filter(x) for x in self)

    def map(self, func):
        """Map a function over the links."""
        return LinksCollection(func(x) for x in self)


class Catalogue:
    def __init__(self, context, hints: dict, source: dict):
        """Initialize the Catalogue.

        Parameters
        ----------
        hints : dict
            The available data information.
        source : dict
            The source to use for requests.
        """
        self.context = context
        self.hints = hints
        self.source = source

    def _build_request(self, link) -> dict:
        vector = link.vector
        #       assert isinstance(vector, ForecastVector), type(vector)
        request = {}
        for k, v in self.source_request.items():
            request[k] = v
        for k, v in link.accumulator.key.items():
            request[k] = v
        step = vector.max - vector.base
        hours = step.total_seconds() // 3600
        minutes = (step.total_seconds() // 60) % 60
        assert minutes == 0, "Only full hours supported in grib/mars"
        request["step"] = int(hours)
        request["date"] = vector.base.strftime("%Y%m%d")
        request["time"] = vector.base.strftime("%H%M")
        return request

    def candidate_intervals(self, current_time: datetime.datetime, **kwargs):
        # If not overridden in subclasses, the informations must be provided in the config
        raise NotImplementedError(f"Available periods must be provided in the config for {self.__class__.__name__}")

    def covering_vectors(self, target) -> list[ForecastVector]:
        if "available_periods" in self.hints:
            candidates = self.hints["available_periods"]
        else:
            candidates = self.candidate_intervals

        coverage = covering_intervals(target.start, target.end, candidates)
        intervals = []
        for i in coverage:
            intervals.append(i)  # ForecastVector(i.start, i.end, base=i.base))

        print(f"  Found covering intervals: for {target}:")
        for c in intervals:
            print(f"    {c}")
        return intervals


def match_fields_to_links(field, links: LinksCollection, debug=False):
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
        if valid_date != link.vector.max:
            dprint(f"  Skipping {link} as valid_date {valid_date} does not match {link.vector.max}")
            continue
        if link.vector.base is not None and base_datetime != link.vector.base:
            dprint(f"  Skipping {link} as base_datetime {base_datetime} does not match {link.vector.base}")
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

    def retrieve_fields(self, links: LinksCollection):
        dates = [link.vector.max for link in links]
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
            intervals.append(ForecastVector(base_datetime + start_step, base_datetime + end_step, base_datetime))
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

    def retrieve_fields(self, links: LinksCollection, debug=False):
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
    candidate_intervals = {
        6: "0-1/0-2/0-3/0-4/0-5/0-6/0-7/0-8/0-9/0-10/0-11/0-12/0-13/0-14/0-15/0-16/0-17/0-18",
        18: "0-1/0-2/0-3/0-4/0-5/0-6/0-7/0-8/0-9/0-10/0-11/0-12/0-13/0-14/0-15/0-16/0-17/0-18",
    }


class EaEndaCatalogue(MarsCatalogue):
    candidate_intervals = {
        6: "0-3/3-6/6-9/9-12/12-15/15-18",
        18: "0-3/3-6/6-9/9-12/12-15/15-18",
    }


class OdOperCatalogue(MarsCatalogue):
    # https://apps.ecmwf.int/mars-catalogue/?stream=oper&levtype=sfc&time=00%3A00%3A00&expver=1&month=aug&year=2020&date=2020-08-25&type=fc&class=od
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
        return {base: steps for base in [0, 3, 6, 9, 12, 15, 18, 21]}


class RrOperCatalogue(MarsCatalogue):
    # https://apps.ecmwf.int/mars-catalogue/?origin=fr-ms-ec&stream=oper&levtype=sfc&time=06%3A00%3A00&expver=prod&month=aug&year=2020&date=2020-08-31&type=fc&class=rr
    # ❌ not tested yet
    candidate_intervals = {i: [(0, 1), (1, 2), (2, 3)] for i in range(0, 22, 3)}


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
        cls = {
            ("ea", "oper"): EaOperCatalogue,
            ("ea", None): EaOperCatalogue,
            ("ea", "enda"): EaEndaCatalogue,
            ("rr", "oper"): RrOperCatalogue,
            ("l5", "oper"): L5OperCatalogue,
            ("od", "oper"): OdOperCatalogue,
            ("od", "enfo"): OdEnfoCatalogue,
            ("od", "elda"): OdEldaCatalogue,
        }[class_, stream]

        return cls(context, hints, source)

    raise ValueError(f"Unknown source_name for catalogue: {source_name}")
