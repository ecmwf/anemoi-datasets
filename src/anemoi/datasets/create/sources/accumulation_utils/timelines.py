# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
from abc import abstractmethod
from typing import Any

import numpy as np
from .utils import TodoList
from numpy.typing import NDArray

class Period:
    value = None

    def __init__(
        self,
        start_datetime: datetime.datetime, 
        end_datetime: datetime.datetime, 
        base_datetime: datetime.datetime
    ):
        assert isinstance(start_datetime, datetime.datetime)
        assert isinstance(end_datetime, datetime.datetime)
        assert isinstance(base_datetime, datetime.datetime)

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        self.base_datetime = base_datetime

    def __eq__(self, other: Any):
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
        assert end_step.total_seconds() % 3600 == 0, end_step  # only full hours supported in grib/mars
        end_step = int(end_step.total_seconds() // 3600)

        return (("date", date), ("time", time), ("step", end_step))
        
    def is_matching_field(self, field):
        
        stepType = field.metadata("stepType")
        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")
        date = field.metadata("date")
        time = field.metadata("time")

        assert stepType == "accum", stepType
        
        base_datetime = datetime.datetime.strptime(str(date) + str(time).zfill(4), "%Y%m%d%H%M")
        start = base_datetime + datetime.timedelta(hours=startStep)
        flag = (start == self.start_datetime)

        end = base_datetime + datetime.timedelta(hours=endStep)
        flag = flag | (end == self.end_datetime)
        
        return flag

    def __repr__(self):
        return f"Period({self.start_datetime} to {self.end_datetime} -> {self.time_request})"

    def length(self):
        return self.end_datetime - self.start_datetime

    def apply(self, accumulated: NDArray | None, values: NDArray) ->NDArray:
        """
        Actual accumulation computation, from a previously accumulated array and a new values array
        """
        if accumulated is None:
            accumulated = np.zeros_like(values)

        assert accumulated.shape == values.shape, (accumulated.shape, values.shape)

        return accumulated + self.sign * values

class Timeline:
    _todo = None

    def __init__(
        self, 
        valid_date: datetime.datetime,
        accumulation_period: datetime.timedelta,
        data_accumulation_period: datetime.timedelta
        ):
        """
        The Timeline object identifies the steps to accumulate on a valid date over an accumulation_period
        This Timeline is made of several consecutive Period objects.
        There is one Timeline object for each accumulated field in the output.
         
        Parameters
        ----------
        valid_date : datetime.datetime
            The valid date for which accumulation is computed
        accumulation_period: datetime.timedelta
            The accumulation period (must be an integer number of hours)
        data_accumulation_period: datetime.timedelta
            The period over which data is already accumulated (must be an integer number of hours)
            data_accumulation_period must be a multiple of accumulation_period
        """
        assert isinstance(valid_date, datetime.datetime), (valid_date, type(valid_date))
        assert isinstance(accumulation_period, datetime.timedelta), (accumulation_period, type(accumulation_period))
        
        self.valid_date = valid_date
        self.accumulation_period = accumulation_period
        self.data_accumulation_period = data_accumulation_period

        assert (self.accumulation_period % self.data_accumulation_period) == datetime.timedelta(
            hours=0
        ), f"accumulation {self.accumulation_period} should be an integer multiple of data_accumulation_period"

        self._periods = self.build_periods()
        self.check_merged_interval()

        self.template_field = None

    def check_merged_interval(self):
        """
        Check the Timeline:
        - made of contiguous chunks
        - each chunk is of length data_accumulation_period
        - the correct start_datetime and end_datetime are defined for each internal period
        """
        global_start = self.valid_date - self.accumulation_period
        global_end = self.valid_date

        interval = np.arange(
            np.datetime64(global_start, "s"), np.datetime64(global_end, "s"), self.data_accumulation_period
        )
        flags = np.zeros_like(interval, dtype=int)
        for p in self._periods:
            segment = np.where((interval >= p.start_datetime) & (interval < p.end_datetime))
            flags[segment] += p.sign
        assert np.all(flags == 1), flags

    def find_matching_period(self, field: Any) -> Period | None:
        """
        For a given field with accumulation, find a period in the Timeline that matches the field metadata
        
        Parameters :
        ----------
        
        field: Any
            The field for which a Period is looked after
            The field originates from the database which reflects the initial dataset
        
        Returns :
        ----------
        
        Period : the corresponding period in the Timeline
        None : if there is no matching period (field outside of Timeline)
        
        """
        found = [p for p in self._periods if p.is_matching_field(field)]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f"Found more than one period for {field}")
        return None

    # flagging logic to check whether all periods in the Timeline have been accumulated
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
        """
        This method will build contiguous Period objects
        Each Period will handle fields and values to accumulate on its own from the previous period.
        """
        pass


class DefaultTimeline(Timeline):
    def __init__(self,
                 valid_date: datetime.datetime,
                 accumulation_period: datetime.timedelta,
                 data_accumulation_period: datetime.timedelta,
                 **kwargs
        ):
        """
        The default Timeline object (as opposed to ERA5/MARS-like periods)
        The base_datetime does depend on the valid time, and is not hard-coded
        IMPORTANT : The default assumption is that the base_datetime for one sample is the datetime of the previous sample in the dataset.
        It can be user-defined if different. For ERA5/MARS-like, the base_datetime is specific and requires different Timelines.
        
        Parameters
        ----------
        valid_date: datetime.datetime

        accumulation_period: datetime.timedelta,

        data_accumulation_period: datetime.timedelta,
      
        """

        self.base_datetime = lambda x: x

        # base datetime can either be user-defined or default to the starting step of accumulation
        if "base_datetime" in kwargs:
            base = int(kwargs["base_datetime"])
            self.base_datetime = lambda x: base

        super().__init__(valid_date, accumulation_period, data_accumulation_period)

    def available_steps(
        self,
        base: datetime.datetime, 
        start: datetime.datetime,
        end: datetime.datetime
    ) -> dict[datetime.datetime, list[datetime.datetime]]:
        """Return the steps available to build/search an available period

        Parameters:
        ----------
            base (int): start of the forecast producing accumulations
            start (int): step (=leadtime) from the forecast where accumulation begins
            end (int): step (=leadtime) from the forecast where accumulation ends
        Returns:
        --------
            _ (dict[List[int]]) :  dictionary listing the available steps between start and end for each base

        """

        list_avail = []
        t = start - base
        while t < (end - base):
            list_avail.append([t, t + self.data_accumulation_period])
            t += self.data_accumulation_period
        return {base: list_avail}

    def search_periods(self, base: datetime.datetime, start: datetime.datetime, end: datetime.datetime, debug=False):
        """
        find candidate periods in the dataset that can be used to accumulate the data
        to get the accumulation between the two dates 'start' and 'end'
        # the periods 
        """
        found = []

        for base_time, steps in self.available_steps(base, start, end).items():

            for step1, step2 in steps:
                if debug:
                    xprint(f"❌ tring: {base_time=} {step1=} {step2=}")

                base_datetime = start - step1

                period = Period(start, end, base_datetime)
                found.append(period)

                assert base_datetime.hour == base_time.hour, (base_datetime, base_time)

                assert period.start_datetime - period.base_datetime == step1, (
                    period.end_datetime,
                    period.base_datetime,
                    step1,
                )
                assert period.end_datetime - period.base_datetime == step2, (
                    period.start_datetime,
                    period.base_datetime,
                    step2,
                )

        return found

    def build_periods(self):
        """
        build the list of periods to accumulate the data
        """

        assert (
            self.base_datetime is not None
        ), "DefaultTimeline needs a base_datetime function, but base_datetime is None"

        lst = []
        for wanted in self.available_steps(
            datetime.timedelta(hours=0), datetime.timedelta(hours=0), self.accumulation_period
        )[datetime.timedelta(hours=0)]:

            start = self.valid_date - wanted[1]
            end = self.valid_date - wanted[0]

            if not end - start == self.data_accumulation_period:
                raise NotImplementedError(f"end and start must be {self.data_accumulation_period} apart")

            found = self.search_periods(self.base_datetime(start), start, end)
            
            if not found:
                xprint(f"❌❌❌ Cannot find accumulation period for {start} {end}")
                self.search_periods(self.base_datetime(start), wanted[0], wanted[1], debug=True)
                raise ValueError(f"Cannot find accumulation period for {start} {end}")

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


class EraTimeline(Timeline):
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
                xprint(f"    Chosing {chosen}, base_datetime: {chosen.base_datetime}")

            chosen.sign = 1

            lst.append(chosen)
        return lst


class EaOperTimeline(EraTimeline):
    def available_steps(self, start, end):
        return {
            6: [[i, i + 1] for i in range(0, 18, 1)],
            18: [[i, i + 1] for i in range(0, 18, 1)],
        }


class L5OperTimeline(EraTimeline):
    def available_steps(self, start, end):
        print("❌❌❌ untested")
        x = 24  # need to check if 24 is the right value
        return {
            0: [[i, i + 1] for i in range(0, x, 1)],
        }


class EaEndaTimeline(EraTimeline):
    def available_steps(self, start, end):
        print("❌❌❌ untested")
        return {
            6: [[i, i + 3] for i in range(0, 18, 1)],
            18: [[i, i + 3] for i in range(0, 18, 1)],
        }


class RrOperTimeline(Timeline):
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


class OdEldaTimeline(EraTimeline):
    def available_steps(self, start, end):
        print("❌❌❌ untested")
        x = 24  # need to check if 24 is the right value
        return {
            6: [[i, i + 1] for i in range(0, x, 1)],
            18: [[i, i + 1] for i in range(0, x, 1)],
        }


class DiffTimeline(Timeline):
    pass


class OdOperTimeline(DiffTimeline):
    def available_steps(self, start, end):
        raise NotImplementedError("need to implement diff and _scda patch (cf accumulation_utils.utils)")


class OdEnfoTimeline(DiffTimeline):
    def available_steps(self, start, end):
        raise NotImplementedError("need to implement diff")


def find_timeline_class(request: dict[str, Any]) -> type[Timeline]:
    try:
        return {
            ("ea", "oper"): EaOperTimeline,  # runs ok
            ("ea", "enda"): EaEndaTimeline,
            ("rr", "oper"): RrOperTimeline,
            ("l5", "oper"): L5OperTimeline,
            ("od", "oper"): OdOperTimeline,
            ("od", "enfo"): OdEnfoTimeline,
            ("od", "elda"): OdEldaTimeline,
        }[request.get("class", None), request.get("stream", None)]

    except KeyError:
        return DefaultTimeline