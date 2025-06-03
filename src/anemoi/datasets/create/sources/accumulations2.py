# (C) Copyright 2024 Anemoi contributors.
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
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import earthkit.data as ekd
import numpy as np
from earthkit.data.core.temporary import temp_file
from earthkit.data.readers.grib.output import new_grib_output
from numpy.typing import NDArray

from anemoi.datasets.create.utils import to_datetime_list

from .legacy import legacy_source

LOG = logging.getLogger(__name__)


xprint = print


def _member(field: Any) -> int:
    """Retrieves the member number from the field metadata.

    Parameters
    ----------
    field : Any
        The field from which to retrieve the member number.

    Returns
    -------
    int
        The member number.
    """
    # Bug in eccodes has number=0 randomly
    number = field.metadata("number", default=0)
    if number is None:
        number = 0
    return number


def _prep_request(request: Dict[str, Any]) -> Dict[str, Any]:
    request = deepcopy(request)

    param = request.pop("param")
    assert isinstance(param, (list, tuple))

    number = request.pop("number", [0])
    if not isinstance(number, (list, tuple)):
        number = [number]

    assert isinstance(number, (list, tuple))
    request["stream"] = request.get("stream", "oper")

    type_ = request.get("type", "an")
    if type_ == "an":
        type_ = "fc"
    request["type"] = type_
    request["levtype"] = request.get("levtype", "sfc")
    if request["levtype"] != "sfc":
        # LOG.warning("'type' should be 'sfc', found %s", request['type'])
        raise NotImplementedError("Only sfc leveltype is supported")

    return request, param, number


class Period:
    value = None

    def __init__(self, start_datetime, end_datetime, base_datetime):
        assert isinstance(start_datetime, datetime.datetime)
        assert isinstance(end_datetime, datetime.datetime)
        assert isinstance(base_datetime, datetime.datetime)

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        self.base_datetime = base_datetime

    def __eq__(self, other):
        if not isinstance(other, Period):
            return False
        return (
            (self.start_datetime == other.start_datetime)
            and (self.end_datetime == other.end_datetime)
            and (self.base_datetime == other.base_datetime)
        )

    def __hash__(self):
        return hash((self.start_datetime, self.end_datetime, self.base_datetime))

    @property
    def time_request(self):
        date = int(self.end_datetime.strftime("%Y%m%d"))
        time = int(self.end_datetime.strftime("%H%M"))

        end_step = self.end_datetime - self.base_datetime
        assert end_step.total_seconds() % 3600 == 0, end_step  # only full hours supported
        end_step = int(end_step.total_seconds() // 3600)

        return (("date", date), ("time", time), ("step", end_step))

    @property
    def time_check(self):
        date = int(self.base_datetime.strftime("%Y%m%d"))
        time = int(self.base_datetime.strftime("%H%M"))

        end_step = self.end_datetime - self.base_datetime
        assert end_step.total_seconds() % 3600 == 0, end_step  # only full hours supported
        end_step = int(end_step.total_seconds() // 3600)

        return (("date", date), ("time", time), ("step", end_step))

    def field_to_key(self, field):
        return (
            ("date", field.metadata("date")),
            ("time", field.metadata("time")),
            ("step", field.metadata("step")),
        )

    def check(self, field):

        stepType = field.metadata("stepType")
        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")
        date = field.metadata("date")
        time = field.metadata("time")

        assert stepType == "accum", stepType

        base_datetime = datetime.datetime.strptime(str(date) + str(time).zfill(4), "%Y%m%d%H%M")

        start = base_datetime + datetime.timedelta(hours=startStep)
        assert start == self.start_datetime, (start, self.start_datetime)

        end = base_datetime + datetime.timedelta(hours=endStep)
        assert end == self.end_datetime, (end, self.end_datetime)

    def is_matching_field(self, field):
        return self.field_to_key(field) == self.time_check

    def __repr__(self):
        return f"Period({self.start_datetime} to {self.end_datetime} -> {self.time_request})"

    def length(self):
        return self.end_datetime - self.start_datetime

    def apply(self, accumulated, values):
        if accumulated is None:
            accumulated = np.zeros_like(values)

        assert accumulated.shape == values.shape, (accumulated.shape, values.shape)

        # if not np.all(values >= 0):
        #     warnings.warn(f"Negative values for {values}: {np.amin(values)} {np.amax(values)}")
        return accumulated + self.sign * values


class TodoList:
    def __init__(self, keys):
        self._todo = set(keys)
        self._len = len(keys)
        self._done = set()
        assert self._len == len(self._todo), (self._len, len(self._todo))

    def is_todo(self, key):
        return key in self._todo

    def is_done(self, key):
        return key in self._done

    def set_done(self, key):
        self._done.add(key)
        self._todo.remove(key)

    def all_done(self):
        if not self._todo:
            assert len(self._done) == self._len, (len(self._done), self._len)
            return True
        return False


class Periods:
    _todo = None

    def __init__(self, valid_date, accumulation_period, **kwargs):
        # one Periods object for each accumulated field in the output

        assert isinstance(valid_date, datetime.datetime), (valid_date, type(valid_date))
        assert isinstance(accumulation_period, datetime.timedelta), (accumulation_period, type(accumulation_period))
        self.valid_date = valid_date
        self.accumulation_period = accumulation_period
        self.kwargs = kwargs

        self._periods = self.build_periods()
        self.check_merged_interval()

        self.template_field = None

    def check_merged_interval(self):
        global_start = self.valid_date - self.accumulation_period
        global_end = self.valid_date
        resolution = datetime.timedelta(hours=1)

        timeline = np.arange(
            np.datetime64(global_start, "s"), np.datetime64(global_end, "s"), np.timedelta64(resolution)
        )

        flags = np.zeros_like(timeline, dtype=int)
        for p in self._periods:
            segment = np.where((timeline >= p.start_datetime) & (timeline < p.end_datetime))
            flags[segment] += p.sign
        assert np.all(flags == 1), flags

    def find_matching_period(self, field):
        # Find a period that matches the field, or return None
        found = [p for p in self._periods if p.is_matching_field(field)]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f"Found more than one period for {field}")
        return None

    def update_template(self, field: Any):
        if self.template_field is None:
            self.template_field = field

        field_date = datetime.datetime.strptime(field.metadata()["valid_datetime"], "%Y-%m-%dT%H:%M:%S")
        template_date = datetime.datetime.strptime(
            self.template_field.metadata()["valid_datetime"], "%Y-%m-%dT%H:%M:%S"
        )

        if field_date < template_date:
            self.template_field = field

    @property
    def todo(self):
        if self._todo is None:
            self._todo = TodoList([p.time_request for p in self._periods])
        return self._todo

    def is_todo(self, period):
        return self.todo.is_todo(period.time_request)

    def is_done(self, period):
        return self.todo.is_done(period.time_request)

    def set_done(self, period):
        self.todo.set_done(period.time_request)

    def all_done(self):
        return self.todo.all_done()

    def __iter__(self):
        return iter(self._periods)

    @abstractmethod
    def build_periods(self):
        pass


class DefaultPeriods(Periods):
    def __init__(self, valid_date, accumulation_period, **kwargs):
        """The default Periods object (as opposed to ERA5/MARS-like periods)
        one Periods object for each accumulated field in the output
        The base_datetime does depend on the valid time, and is not hard-coded
        The default assumption is that the base datetime for one sample is the datetime of the previous sample in the dataset.
        It can be user-defined if different
        """

        self.base_datetime = lambda x: x

        # base datetime can either be user-defined or default to the starting step of accumulation
        if "base_datetime" in kwargs.keys():
            base = int(kwargs["base_datetime"])
            self.base_datetime = lambda x: base

        if "data_accumulation" in kwargs.keys():
            self.data_accumulation_period = int(kwargs.pop("data_accumulation_period"))
        else:
            self.data_accumulation_period = 1

        super().__init__(valid_date, accumulation_period, **kwargs)

    def available_steps(
        self, base: datetime.datetime, start: datetime.datetime, end: datetime.datetime
    ) -> Dict[int, List[int]]:
        """Return the steps available to build/search an available period

        Arguments:
            base (int): start of the forecast producing accumulations
            start (int): step (=leadtime) from the forecast where accumulation begins
            end (int): step (=leadtime) from the forecast where accumulation ends
        Returns:
            _ (Dict[List[int]]) :  dictionary listing the available steps between start and end for each base

        """

        start_base_delta = int((start - base).total_seconds() // 3600)
        end_start_delta = int((end - start).total_seconds() // 3600)
        return {
            base: [
                [i, i + self.data_accumulation_period]
                for i in range(start_base_delta, start_base_delta + end_start_delta, self.data_accumulation_period)
            ],
        }

    def search_periods(self, base: datetime.datetime, start: datetime.datetime, end: datetime.datetime, debug=False):
        # find candidate periods that can be used to accumulate the data
        # to get the accumulation between the two dates 'start' and 'end'
        found = []

        for base_time, steps in self.available_steps(base, start, end).items():

            for step1, step2 in steps:
                if debug:
                    xprint(f"❌ tring: {base_time=} {step1=} {step2=}")

                base_datetime = start - datetime.timedelta(hours=step1)

                period = Period(start, end, base_datetime)
                found.append(period)

                assert base_datetime.hour == base_time.hour, (base_datetime, base_time)

                assert period.start_datetime - period.base_datetime == datetime.timedelta(hours=step1), (
                    period.end_datetime,
                    period.base_datetime,
                    step1,
                )
                assert period.end_datetime - period.base_datetime == datetime.timedelta(hours=step2), (
                    period.start_datetime,
                    period.base_datetime,
                    step2,
                )

        return found

    def build_periods(self):
        # build the list of periods to accumulate the data

        hours = self.accumulation_period.total_seconds() / 3600
        assert int(hours) == hours, f"Only full hours accumulation is supported {hours}"
        hours = int(hours)

        assert (
            self.base_datetime is not None
        ), "DefaultPeriods needs a base_datetime function, but base_datetime is None"

        lst = []
        for wanted in [[i, i + self.data_accumulation_period] for i in range(0, hours, self.data_accumulation_period)]:

            start = self.valid_date - datetime.timedelta(hours=wanted[1])
            end = self.valid_date - datetime.timedelta(hours=wanted[0])

            if not end - start == datetime.timedelta(hours=self.data_accumulation_period):
                raise NotImplementedError(f"end and start must be {self.data_accumulation_period} hours apart")

            found = self.search_periods(self.base_datetime(start), start, end)
            if not found:
                xprint(f"❌❌❌ Cannot find accumulation for {start} {end}")
                self.search_periods(self.base_datetime(start), wanted[0], wanted[1], debug=True)
                raise ValueError(f"Cannot find accumulation for {start} {end}")

            found = sorted(found, key=lambda x: x.base_datetime, reverse=True)
            chosen = found[0]

            if len(found) > 1:
                xprint(f"  Found more than one period for {start} {end}")
                for f in found:
                    xprint(f"    {f}")
                xprint(f"    Chosing {chosen}")

            chosen.sign = 1

            lst.append(chosen)
        return lst


class EraPeriods(Periods):
    def search_periods(self, start, end, debug=False):
        # find candidate periods that can be used to accumulate the data
        # to get the accumulation between the two dates 'start' and 'end'
        found = []
        if not end - start == datetime.timedelta(hours=1):
            raise NotImplementedError("Only 1 hour period is supported")

        for base_time, steps in self.available_steps(start, end).items():
            for step1, step2 in steps:
                if debug:
                    xprint(f"❌ tring: {base_time=} {step1=} {step2=}")

                if ((base_time + step1) % 24) != start.hour:
                    continue

                if ((base_time + step2) % 24) != end.hour:
                    continue

                base_datetime = start - datetime.timedelta(hours=step1)

                period = Period(start, end, base_datetime)
                found.append(period)

                assert base_datetime.hour == base_time, (base_datetime, base_time)

                assert period.start_datetime - period.base_datetime == datetime.timedelta(hours=step1), (
                    period.start_datetime,
                    period.base_datetime,
                    step1,
                )
                assert period.end_datetime - period.base_datetime == datetime.timedelta(hours=step2), (
                    period.end_datetime,
                    period.base_datetime,
                    step2,
                )

        return found

    def build_periods(self):
        # build the list of periods to accumulate the data

        hours = self.accumulation_period.total_seconds() / 3600
        assert int(hours) == hours, f"Only full hours accumulation is supported {hours}"
        hours = int(hours)

        lst = []
        for wanted in [[i, i + 1] for i in range(0, hours, 1)]:

            start = self.valid_date - datetime.timedelta(hours=wanted[1])
            end = self.valid_date - datetime.timedelta(hours=wanted[0])

            found = self.search_periods(start, end)
            if not found:
                xprint(f"❌❌❌ Cannot find accumulation for {start} {end}")
                self.search_periods(start, end, debug=True)
                raise ValueError(f"Cannot find accumulation for {start} {end}")

            found = sorted(found, key=lambda x: x.base_datetime, reverse=True)
            chosen = found[0]

            if len(found) > 1:
                xprint(f"  Found more than one period for {start} {end}")
                for f in found:
                    xprint(f"    {f}")
                xprint(f"    Chosing {chosen}")

            chosen.sign = 1

            lst.append(chosen)
        return lst


class EaOperPeriods(EraPeriods):
    def available_steps(self, start, end):
        return {
            6: [[i, i + 1] for i in range(0, 18, 1)],
            18: [[i, i + 1] for i in range(0, 18, 1)],
        }


class L5OperPeriods(EraPeriods):
    def available_steps(self, start, end):
        print("❌❌❌ untested")
        x = 24  # need to check if 24 is the right value
        return {
            0: [[i, i + 1] for i in range(0, x, 1)],
        }


class EaEndaPeriods(EraPeriods):
    def available_steps(self, start, end):
        print("❌❌❌ untested")
        return {
            6: [[i, i + 3] for i in range(0, 18, 1)],
            18: [[i, i + 3] for i in range(0, 18, 1)],
        }


class RrOperPeriods(Periods):
    def available_steps(self, start, end):
        raise NotImplementedError("need to implement diff")
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


class OdEldaPeriods(EraPeriods):
    def available_steps(self, start, end):
        print("❌❌❌ untested")
        x = 24  # need to check if 24 is the right value
        return {
            6: [[i, i + 1] for i in range(0, x, 1)],
            18: [[i, i + 1] for i in range(0, x, 1)],
        }


class DiffPeriods(Periods):
    pass


class OdOperPeriods(DiffPeriods):
    def available_steps(self, start, end):
        raise NotImplementedError("need to implement diff and _scda patch")


class OdEnfoPeriods(DiffPeriods):
    def available_steps(self, start, end):
        raise NotImplementedError("need to implement diff")


def find_accumulator_class(request: Dict[str, Any]) -> Periods:

    try:
        return {
            ("ea", "oper"): EaOperPeriods,  # runs ok
            ("ea", "enda"): EaEndaPeriods,
            ("rr", "oper"): RrOperPeriods,
            ("l5", "oper"): L5OperPeriods,
            ("od", "oper"): OdOperPeriods,
            ("od", "enfo"): OdEnfoPeriods,
            ("od", "elda"): OdEldaPeriods,
        }[request.get("class", None), request.get("stream", None)]

    except KeyError:
        return DefaultPeriods


class Accumulator:
    values = None

    def __init__(self, period_class, out, valid_date, user_accumulation_period, **kwargs):
        self.valid_date = valid_date

        # keep the reference to the output file to be able to write the result using an input field as template
        self.out = out

        # key contains the mars request parameters except the one related to the time
        # A mars request is a dictionary with three categories of keys:
        #   - the ones related to the time (date, time, step)
        #   - the ones related to the data (param, stream, levtype, expver, number, ...)
        #   - the ones related to the processing to be done (grid, area, ...)
        self.kwargs = kwargs
        for k in ["date", "time", "step"]:
            if k in kwargs:
                raise ValueError(f"Cannot use {k} in kwargs for accumulations")

        self.key = {k: v for k, v in kwargs.items() if k in ["param", "level", "levelist", "number"]}

        self.periods = period_class(self.valid_date, user_accumulation_period, **kwargs)

    @property
    def requests(self) -> Dict:
        for period in self.periods:
            # build the full data requests, merging the time requests with the key
            yield {**self.kwargs.copy(), **dict(period.time_request)}

    def is_field_needed(self, field):
        for k, v in self.key.items():
            if field.metadata(k, default=0) != v:
                LOG.debug(f"{self} does not need field {field} because of {k}={field.metadata(k,default=0)} not {v}")
                return False
        return True

    def compute(self, field: Any, values: NDArray) -> None:
        """Verify the field time metadata, find the associated period
         and perform accumulation with the given values

        in any case values have been exrtacted from field

        """

        # check if field has correct parameters for the Accumulator (param, number)
        if not self.is_field_needed(field):
            return

        period = self.periods.find_matching_period(field)
        if not period:
            return

        assert self.periods.is_todo(period), (self.periods, period)
        assert not self.periods.is_done(period), f"Field {field} for period {period} already done"

        period.check(field)

        # template field for write must be updated with the oldest field encountered for the accumulator
        self.periods.update_template(field)

        xprint(f"{self} field ✅ ({period.sign}){field} for {period}")

        self.values = period.apply(self.values, values)
        self.periods.set_done(period)

        if self.periods.all_done():
            # all periods for accumulation have been processed
            # template field is now the oldest
            # temporary file can be written
            self.write()
            xprint("accumulator", self, " : data written ✅ ")

    def write(self) -> None:
        assert self.periods.all_done(), self.periods

        if np.all(self.values < 0):
            LOG.warning(
                f"Negative values when computing accumutation for {self}): min={np.amin(self.values)} max={np.amax(self.values)}"
            )

        startStep = 0
        endStep = self.periods.accumulation_period.total_seconds() // 3600
        assert int(endStep) == endStep, "only full hours accumulation is supported"
        endStep = int(endStep)
        fake_base_date = self.valid_date - self.periods.accumulation_period
        date = int(fake_base_date.strftime("%Y%m%d"))
        time = int(fake_base_date.strftime("%H%M"))

        self.out.write(
            self.values,
            template=self.periods.template_field,
            stepType="accum",
            startStep=startStep,
            endStep=endStep,
            date=date,
            time=time,
            check_nans=True,
        )
        self.values = None

    def __repr__(self):
        key = ", ".join(f"{k}={v}" for k, v in self.key.items())
        return f"{self.__class__.__name__}({self.valid_date}, {key})"


def _compute_accumulations(
    context: Any,
    dates: List[datetime.datetime],
    action: Any,
    request: Dict[str, Any],
    user_accumulation_period: datetime.timedelta,
    # data_accumulation_period: Optional[int] = None,
    # patch: Any = _identity,
) -> Any:

    request, param, number = _prep_request(request)

    period_class = find_accumulator_class(request)

    tmp = temp_file()
    path = tmp.path
    out = new_grib_output(path)

    # build one accumulator per output field
    accumulators = []
    overlapping_periods = set()

    for valid_date in dates:
        for p in param:
            for n in number:
                accumulators.append(
                    Accumulator(
                        period_class,
                        out,
                        valid_date,
                        user_accumulation_period=user_accumulation_period,
                        param=p,
                        number=n,
                        **request,
                    )
                )

        # this is the exact number of periods that should be retrieved from action.source
        overlapping_periods.update({period for period in accumulators[-1].periods})

    # get all needed data requests
    requests = []
    for a in accumulators:
        for r in a.requests:
            requests.append(r)

    # get the data (this will pack the requests to avoid duplicates and make a minimal number of requests)
    action.source.kwargs = {"request_already_using_valid_datetime": True, "shift_time_request": True}
    action.source.args = requests
    action.source.context = context
    ds = action.source.execute(dates)

    assert len(ds) / len(param) / len(number) == len(overlapping_periods), (
        f"retrieval yields {len(ds)} fields, {len(param)} params, {len(number)} members ",
        f"but total number of periods requested is {len(overlapping_periods)}",
        f"❌❌❌ error in {action.source}",
    )

    # send each field to the each accumulator, the accumulator will use the field to the accumulation
    # if the accumulator has requested it

    for field in ds:
        values = field.values  # optimisation : reading values only once
        for a in accumulators:
            a.compute(field, values)

    out.close()

    ds = ekd.from_source("file", path)

    assert len(ds) / len(param) / len(number) == len(dates), (
        len(ds),
        len(param),
        len(dates),
    )

    # keep a reference to the tmp file, or it gets deleted when the function returns
    ds._tmp = tmp

    return ds


def _to_list(x: Union[List[Any], Tuple[Any], Any]) -> List[Any]:
    """Converts the input to a list if it is not already a list or tuple.

    Parameters
    ----------
    x : Union[List[Any], Tuple[Any], Any]
        Input value.

    Returns
    -------
    List[Any]
        The input value as a list.
    """
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def _scda(request: Dict[str, Any]) -> Dict[str, Any]:
    """Modifies the request stream based on the time.

    Parameters
    ----------
    request : Dict[str, Any]
        Request parameters.

    Returns
    -------
    Dict[str, Any]
        The modified request parameters.
    """
    if request["time"] in (6, 18, 600, 1800):
        request["stream"] = "scda"
    else:
        request["stream"] = "oper"
    return request


@legacy_source(__file__)
def accumulations(context, dates, action, **request):
    _to_list(request["param"])
    try:
        user_accumulation_period = request.pop("accumulation_period")
    except KeyError:
        raise ValueError("Accumulate action should provide 'accumulation_period', but none was found.")

    user_accumulation_period = datetime.timedelta(hours=user_accumulation_period)

    context.trace("🌧️", f"accumulations {request} {user_accumulation_period}")

    return _compute_accumulations(
        context,
        dates,
        action,
        request,
        user_accumulation_period=user_accumulation_period,
    )


execute = accumulations

if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """
      class: ea
      expver: '0001'
      grid: 20./20.
      levtype: sfc
#      number: [0, 1]
#      stream: enda
      param: [cp, tp]
#      accumulation_period: 6h
      accumulation_period: 2
    """
    )
    dates = yaml.safe_load("[2022-12-31 00:00, 2022-12-31 06:00]")
    # dates = yaml.safe_load("[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]")
    dates = to_datetime_list(dates)

    class Context:
        use_grib_paramid = True

        def trace(self, *args):
            print(*args)

    for f in accumulations(Context, dates, **config):
        print(f, f.to_numpy().mean())
