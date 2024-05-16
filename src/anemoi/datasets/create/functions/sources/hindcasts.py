# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import warnings
from copy import deepcopy

import climetlab as cml
import numpy as np
from climetlab.core.temporary import temp_file
from climetlab.readers.grib.output import new_grib_output
from climetlab.utils.availability import Availability

from anemoi.datasets.create.functions.sources.mars import mars

DEBUG = True


def member(field):
    # Bug in eccodes has number=0 randomly
    number = field.metadata("number")
    if number is None:
        number = 0
    return number


class Accumulation:
    def __init__(self, out, /, param, date, time, number, step, frequency, **kwargs):
        self.out = out
        self.param = param
        self.date = date
        self.time = time
        self.steps = step
        self.number = number
        self.values = None
        self.seen = set()
        self.startStep = None
        self.endStep = None
        self.done = False
        self.frequency = frequency
        self._check = None

    @property
    def key(self):
        return (self.param, self.date, self.time, self.steps, self.number)

    def check(self, field):
        if self._check is None:
            self._check = field.as_mars()

            assert self.param == field.metadata("param"), (self.param, field.metadata("param"))
            assert self.date == field.metadata("date"), (self.date, field.metadata("date"))
            assert self.time == field.metadata("time"), (self.time, field.metadata("time"))
            assert self.number == member(field), (self.number, member(field))

            return

        mars = field.as_mars()
        keys1 = sorted(self._check.keys())
        keys2 = sorted(mars.keys())

        assert keys1 == keys2, (keys1, keys2)

        for k in keys1:
            if k not in ("step",):
                assert self._check[k] == mars[k], (k, self._check[k], mars[k])

    def write(self, template):

        assert self.startStep != self.endStep, (self.startStep, self.endStep)
        assert np.all(self.values >= 0), (np.amin(self.values), np.amax(self.values))

        self.out.write(
            self.values,
            template=template,
            stepType="accum",
            startStep=self.startStep,
            endStep=self.endStep,
        )
        self.values = None
        self.done = True

    def add(self, field, values):

        self.check(field)

        step = field.metadata("step")
        if step not in self.steps:
            return

        if not np.all(values >= 0):
            warnings.warn(f"Negative values for {field}: {np.amin(values)} {np.amax(values)}")

        assert not self.done, (self.key, step)
        assert step not in self.seen, (self.key, step)

        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")

        if self.buggy_steps and startStep == endStep:
            startStep = 0

        assert step == endStep, (startStep, endStep, step)

        self.compute(values, startStep, endStep)

        self.seen.add(step)

        if len(self.seen) == len(self.steps):
            self.write(template=field)

    @classmethod
    def mars_date_time_steps(cls, dates, step1, step2, frequency, base_times, adjust_step):

        # assert step1 > 0, (step1, step2, frequency)

        for valid_date in dates:
            base_date = valid_date - datetime.timedelta(hours=step2)
            add_step = 0
            if base_date.hour not in base_times:
                if not adjust_step:
                    raise ValueError(
                        f"Cannot find a base time in {base_times} that validates on {valid_date.isoformat()} for step={step2}"
                    )

                while base_date.hour not in base_times:
                    # print(f'{base_date=}, {base_times=}, {add_step=} {frequency=}')
                    base_date -= datetime.timedelta(hours=1)
                    add_step += 1

            yield cls._mars_date_time_step(base_date, step1, step2, add_step, frequency)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.key})"


class AccumulationFromStart(Accumulation):
    buggy_steps = True

    def compute(self, values, startStep, endStep):

        assert startStep == 0, startStep

        if self.values is None:

            self.values = np.copy(values)
            self.startStep = 0
            self.endStep = endStep

        else:
            assert endStep != self.endStep, (self.endStep, endStep)

            if endStep > self.endStep:
                # assert endStep - self.endStep == self.stepping, (self.endStep, endStep, self.stepping)
                self.values = values - self.values
                self.startStep = self.endStep
                self.endStep = endStep
            else:
                # assert self.endStep - endStep == self.stepping, (self.endStep, endStep, self.stepping)
                self.values = self.values - values
                self.startStep = endStep

            if not np.all(self.values >= 0):
                warnings.warn(f"Negative values for {self.param}: {np.amin(self.values)} {np.amax(self.values)}")
                self.values = np.maximum(self.values, 0)

    @classmethod
    def _mars_date_time_step(cls, base_date, step1, step2, add_step, frequency):
        assert not frequency, frequency

        steps = (step1 + add_step, step2 + add_step)
        if steps[0] == 0:
            steps = (steps[1],)

        return (
            base_date.year * 10000 + base_date.month * 100 + base_date.day,
            base_date.hour * 100 + base_date.minute,
            steps,
        )


class AccumulationFromLastStep(Accumulation):
    buggy_steps = False

    def compute(self, values, startStep, endStep):

        assert endStep - startStep == self.frequency, (startStep, endStep, self.frequency)

        if self.startStep is None:
            self.startStep = startStep
        else:
            self.startStep = min(self.startStep, startStep)

        if self.endStep is None:
            self.endStep = endStep
        else:
            self.endStep = max(self.endStep, endStep)

        if self.values is None:
            self.values = np.zeros_like(values)

        self.values += values

    @classmethod
    def _mars_date_time_step(cls, base_date, step1, step2, add_step, frequency):
        assert frequency > 0, frequency
        # assert step1 > 0, (step1, step2, frequency, add_step, base_date)

        steps = []
        for step in range(step1 + frequency, step2 + frequency, frequency):
            steps.append(step + add_step)
        return (
            base_date.year * 10000 + base_date.month * 100 + base_date.day,
            base_date.hour * 100 + base_date.minute,
            tuple(steps),
        )


def identity(x):
    return x


def compute_accumulations(
    dates,
    request,
    user_accumulation_period=6,
    data_accumulation_period=None,
    patch=identity,
    base_times=None,
):
    adjust_step = isinstance(user_accumulation_period, int)

    if not isinstance(user_accumulation_period, (list, tuple)):
        user_accumulation_period = (0, user_accumulation_period)

    assert len(user_accumulation_period) == 2, user_accumulation_period
    step1, step2 = user_accumulation_period
    assert step1 < step2, user_accumulation_period

    if base_times is None:
        base_times = [0, 6, 12, 18]

    base_times = [t // 100 if t > 100 else t for t in base_times]

    AccumulationClass = AccumulationFromStart if data_accumulation_period in (0, None) else AccumulationFromLastStep

    mars_date_time_steps = AccumulationClass.mars_date_time_steps(
        dates,
        step1,
        step2,
        data_accumulation_period,
        base_times,
        adjust_step,
    )

    request = deepcopy(request)

    param = request["param"]
    if not isinstance(param, (list, tuple)):
        param = [param]

    number = request.get("number", [0])
    assert isinstance(number, (list, tuple))

    frequency = data_accumulation_period

    type_ = request.get("type", "an")
    if type_ == "an":
        type_ = "fc"

    request.update({"type": type_, "levtype": "sfc"})

    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    requests = []

    accumulations = {}

    for date, time, steps in mars_date_time_steps:
        for p in param:
            for n in number:
                requests.append(
                    patch(
                        {
                            "param": p,
                            "date": date,
                            "time": time,
                            "step": sorted(steps),
                            "number": n,
                        }
                    )
                )

    compressed = Availability(requests)
    ds = cml.load_source("empty")
    for r in compressed.iterate():
        request.update(r)
        print("🌧️", request)
        ds = ds + cml.load_source("mars", **request)

    accumulations = {}
    for a in [AccumulationClass(out, frequency=frequency, **r) for r in requests]:
        for s in a.steps:
            key = (a.param, a.date, a.time, s, a.number)
            accumulations.setdefault(key, []).append(a)

    for field in ds:
        key = (
            field.metadata("param"),
            field.metadata("date"),
            field.metadata("time"),
            field.metadata("step"),
            member(field),
        )
        values = field.values  # optimisation
        assert accumulations[key], key
        for a in accumulations[key]:
            a.add(field, values)

    for acc in accumulations.values():
        for a in acc:
            assert a.done, (a.key, a.seen, a.steps)

    out.close()

    ds = cml.load_source("file", path)

    assert len(ds) / len(param) / len(number) == len(dates), (
        len(ds),
        len(param),
        len(dates),
    )
    ds._tmp = tmp

    return ds


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def normalise_time_to_hours(r):
    r = deepcopy(r)
    if "time" not in r:
        return r

    times = []
    for t in to_list(r["time"]):
        assert len(t) == 4, r
        assert t.endswith("00"), r
        times.append(int(t) // 100)
    r["time"] = tuple(times)
    return r


def normalise_number(r):
    if "number" not in r:
        return r
    number = r["number"]
    number = to_list(number)

    if len(number) > 4 and (number[1] == "to" and number[3] == "by"):
        return list(range(int(number[0]), int(number[2]) + 1, int(number[4])))

    if len(number) > 2 and number[1] == "to":
        return list(range(int(number[0]), int(number[2]) + 1))

    r["number"] = number
    return r


class HindcastCompute:
    def __init__(self, base_times, available_steps, request):
        self.base_times = base_times
        self.available_steps = available_steps
        self.request = request

    def compute_hindcast(self, date):
        for step in self.available_steps:
            start_date = date - datetime.timedelta(hours=step)
            hours = start_date.hour
            if hours in self.base_times:
                r = deepcopy(self.request)
                r["date"] = start_date
                r["time"] = f"{start_date.hour:02d}00"
                r["step"] = step
                return r
        raise ValueError(
            f"Cannot find data for {self.request} for {date} (base_times={self.base_times}, available_steps={self.available_steps})"
        )


def use_reference_year(reference_year, request):
    request = deepcopy(request)
    hdate = request.pop("date")
    date = datetime.datetime(reference_year, hdate.month, hdate.day)
    request.update(date=date.strftime("%Y-%m-%d"), hdate=hdate.strftime("%Y-%m-%d"))
    return request


def hindcasts(context, dates, **request):
    request["param"] = to_list(request["param"])
    request["step"] = to_list(request["step"])
    request["step"] = [int(_) for _ in request["step"]]

    if request.get("stream") == "enfh" and "base_times" not in request:
        request["base_times"] = [0]

    available_steps = request.pop("step")
    available_steps = to_list(available_steps)

    base_times = request.pop("base_times")

    reference_year = request.pop("reference_year")

    context.trace("H️", f"hindcast {request} {base_times} {available_steps} {reference_year}")

    c = HindcastCompute(base_times, available_steps, request)
    requests = []
    for d in dates:
        req = c.compute_hindcast(d)
        req = use_reference_year(reference_year, req)

        requests.append(req)
    return mars(context, dates, *requests, date_key="hdate")


execute = hindcasts
