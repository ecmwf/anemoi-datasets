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
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray


class Interval:
    sign: int | None = None

    def __init__(
        self, start_datetime: datetime.datetime, end_datetime: datetime.datetime, base_datetime: datetime.datetime
    ):
        """Defining an insecable time segment carrying all temporal information.

        Parameters:
        ----------
        start_datetime: datetime.datetime
            the date and hour at which the inte starts
        end_datetime: datetime.datetime
            the date and hour at which the interval ends
        base_datetime: datetime.datetime
            the date and hour referring to the beginning of the forecast run that has produced the data
            At this base_datetime, the accumulation of the forecast is zero.
            In the case of DefaultIntervalsCollections, this is equal to start_datetime.
        """
        assert isinstance(start_datetime, datetime.datetime)
        assert isinstance(end_datetime, datetime.datetime)
        assert isinstance(base_datetime, datetime.datetime)

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        self.base_datetime = base_datetime

        assert base_datetime <= start_datetime, "invalid base/start dates for Interval"
        assert start_datetime <= end_datetime, "invalid start/end dates for Interval"

    def __eq__(self, other: Any):
        if not isinstance(other, Interval):
            return False
        return (
            (self.start_datetime == other.start_datetime)
            and (self.end_datetime == other.end_datetime)
            and (self.base_datetime == other.base_datetime)
        )

    def __hash__(self):
        return hash((self.start_datetime, self.end_datetime, self.base_datetime))

    @property
    def time_request(self) -> tuple:
        """Create a formatted time request for the database."""
        date = int(self.end_datetime.strftime("%Y%m%d"))
        time = int(self.end_datetime.strftime("%H%M"))

        end_step = self.end_datetime - self.base_datetime
        assert end_step.total_seconds() % 3600 == 0, end_step  # only full hours supported in grib/mars
        end_step = int(end_step.total_seconds() // 3600)

        return (("date", date), ("time", time), ("step", end_step))

    def is_matching_field(self, field: Any):
        """Check whether a given field matches the current Interval
        in terms of timestamps.
        """
        stepType = field.metadata("stepType")
        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")
        date = field.metadata("date")
        time = field.metadata("time")

        assert stepType == "accum", f"Not an accumulated variable: {stepType}"

        base_datetime = datetime.datetime.strptime(str(date) + str(time).zfill(4), "%Y%m%d%H%M")
        start = base_datetime + datetime.timedelta(hours=startStep)
        flag = start == self.start_datetime

        end = base_datetime + datetime.timedelta(hours=endStep)
        flag = flag and (end == self.end_datetime)
        return flag

    def __repr__(self):
        return f"Interval({self.start_datetime} to {self.end_datetime} -> {self.time_request}, counted {self.sign})"

    def length(self):
        """Time length of the interval"""
        return self.end_datetime - self.start_datetime

    def apply(self, accumulated: NDArray | None, values: NDArray) -> NDArray:
        """Actual accumulation computation, from a previously accumulated array and a new values array"""
        if accumulated is None:
            accumulated = np.zeros_like(values)

        assert accumulated.shape == values.shape, (accumulated.shape, values.shape)

        return accumulated + self.sign * values


class IntervalsCollection:
    _todo: set | None = None

    def __init__(
        self,
        valid_date: datetime.datetime,
        accumulation_period: datetime.timedelta,
        data_accumulation_period: datetime.timedelta,
        **kwargs: dict,
    ):
        """The IntervalsCollection object identifies the steps to accumulate on a valid date over an accumulation_period
        This IntervalsCollection is made of several consecutive Interval objects.
        There is one IntervalsCollection object for each accumulated field in the output.

        Parameters:
        ----------
        valid_date : datetime.datetime
            The valid date for which accumulation is computed
        accumulation_period: datetime.timedelta
            The accumulation interval (must be an integer number of hours)
        data_accumulation_period: datetime.timedelta
            The interval over which data is already accumulated (must be an integer number of hours)
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

        self._intervals = self.build_intervals()
        self.check_merged_interval()

        self.template_field = None

    def check_merged_interval(self):
        """Check the IntervalsCollection:
        - made of contiguous chunks
        - each chunk is of length data_accumulation_period
        - the correct start_datetime and end_datetime are defined for each internal interval
        """
        global_start = self.valid_date - self.accumulation_period
        global_end = self.valid_date

        interval = np.arange(
            np.datetime64(global_start, "s"), np.datetime64(global_end, "s"), self.data_accumulation_period
        )
        flags = np.zeros_like(interval, dtype=int)
        for p in self._intervals:
            segment = np.where((interval >= p.start_datetime) & (interval < p.end_datetime))
            flags[segment] += p.sign
        assert np.all(flags == 1), flags

    def find_matching_interval(self, field: Any) -> Interval | None:
        """For a given field with accumulation, find a interval in the IntervalsCollection that matches the field metadata

        Parameters
        ----------

        field: Any
            The field for which a Interval is looked after
            The field originates from the database which reflects the initial dataset

        Return
        ------

        Interval : the corresponding interval in the IntervalsCollection
        None : if there is no matching interval (field outside of IntervalsCollection)

        """
        found = [p for p in self._intervals if p.is_matching_field(field)]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f"Found more than one interval for {field}")
        return None

    # flagging logic to check whether all intervals in the IntervalsCollection have been accumulated
    @property
    def todo(self):
        if self._todo is None:
            self._todo = set([p.time_request for p in self._intervals])
            self._len = len(self._todo)
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
        return self._todo

    def __iter__(self):
        return iter(self._intervals)

    @abstractmethod
    def build_intervals(self):
        """This method will build contiguous Interval objects
        Each Interval will handle fields and values to accumulate on its own from the previous interval.
        """
        pass


class DefaultIntervalsCollection(IntervalsCollection):
    def __init__(
        self,
        valid_date: datetime.datetime,
        accumulation_period: datetime.timedelta,
        data_accumulation_period: datetime.timedelta,
        **kwargs: dict,
    ):
        """The default IntervalsCollection object (as opposed to ERA5/MARS-like intervals)
        The base_datetime does depend on the valid time, and is not hard-coded
        IMPORTANT : The default assumption is that the base_datetime for one sample is the datetime of the previous sample in the dataset.
        It can be user-defined if different. For ERA5/MARS-like, the base_datetime is specific and requires different IntervalsCollections.

        Parameters
        ----------

        valid_date: datetime.datetime
            Validity date of the IntervalsCollection
        accumulation_period: datetime.timedelta,
            User-defined accumulation interval
        data_accumulation_period: datetime.timedelta
            Source data accumulation interval
        kwargs: dict
            Additional parameters, among which an optional base_datetime: str
        """

        self.base_datetime = lambda x: x

        # base datetime can either be user-defined or default to the starting step of accumulation
        if "base_datetime" in kwargs:
            base = frequency_to_timedelta(kwargs["base_datetime"])
            self.base_datetime = lambda x: base

        super().__init__(valid_date, accumulation_period, data_accumulation_period)

    def available_steps(
        self, base: datetime.datetime, start: datetime.datetime, end: datetime.datetime
    ) -> dict[datetime.datetime, list[datetime.datetime]]:
        """Return the IntervalsCollection time steps available to build/search an available interval

        Parameters:
        ----------
            base (datetime.datetime): start of the forecast producing accumulations
            start (datetime.datetime): step (=leadtime) from the forecast where interval begins
            end (datetime.datetime): step (=leadtime) from the forecast where interval ends

        Return:
        -------
            _ (dict[List[int]]) :  dictionary listing the available steps between start and end for each base

        """

        list_avail = []
        t = start - base
        while t < (end - base):
            list_avail.append([t, t + self.data_accumulation_period])
            t += self.data_accumulation_period
        return {base: list_avail}

    def build_intervals(self) -> list[Interval]:
        """Build the list of intervals to accumulate the data on the IntervalsCollection"""

        assert (
            self.base_datetime is not None
        ), "DefaultIntervalsCollection needs a base_datetime function, but base_datetime is None"

        lst = []
        for wanted in self.available_steps(
            datetime.timedelta(hours=0), datetime.timedelta(hours=0), self.accumulation_period
        )[datetime.timedelta(hours=0)]:

            start = self.valid_date - wanted[1]
            end = self.valid_date - wanted[0]

            if not end - start == self.data_accumulation_period:
                raise NotImplementedError(f"end and start must be {self.data_accumulation_period} apart")

            found = Interval(start, end, self.base_datetime(start))

            if not found:
                print(f"❌❌❌ Cannot find accumulation interval for {start} {end}")
                self.search_intervals(self.base_datetime(start), wanted[0], wanted[1], debug=True)
                raise ValueError(f"Cannot find accumulation interval for {start} {end}")

            found.sign = 1

            lst.append(found)
        return lst


class EraIntervalsCollection(IntervalsCollection):
    def search_intervals(self, start: datetime.datetime, end: datetime.datetime, debug: bool = False) -> list[Interval]:
        """Find candidate intervals that can be used to accumulate the data
        between the two dates 'start' and 'end'.
        Depending on the ERA configuration, one might have several corresponding intervals with the same dates.

        Parameters
        ----------
        start: datetime.datetime
            starting date for the interval
        end: datetime.datetime
            end date for the interval
        debug: bool, default
            print debug information

        Return
        ------

        found: list[Interval]
            The list of matching intervals
        """
        found = []
        if not end - start == datetime.timedelta(hours=1):
            raise NotImplementedError("Only 1 hour interval is supported")

        for base_time, steps in self.available_steps(start, end).items():
            for step1, step2 in steps:
                if debug:
                    print(f"❌ tring: {base_time=} {step1=} {step2=}")

                if ((base_time + step1) % 24) != start.hour:
                    continue

                if ((base_time + step2) % 24) != end.hour:
                    continue

                base_datetime = start - datetime.timedelta(hours=step1)
                interval = Interval(start, end, base_datetime)
                found.append(interval)

                assert base_datetime.hour == base_time, (base_datetime, base_time)

                assert interval.start_datetime - interval.base_datetime == datetime.timedelta(hours=step1), (
                    interval.start_datetime,
                    interval.base_datetime,
                    step1,
                )
                assert interval.end_datetime - interval.base_datetime == datetime.timedelta(hours=step2), (
                    interval.end_datetime,
                    interval.base_datetime,
                    step2,
                )

        return found

    def build_intervals(self) -> list[Interval]:
        """Build the list of intervals to accumulate the data on the IntervalsCollection"""

        hours = self.accumulation_period.total_seconds() / 3600
        assert int(hours) == hours, f"Only full hours accumulation is supported {hours}"
        hours = int(hours)

        lst = []
        for wanted in [[i, i + 1] for i in range(0, hours, 1)]:

            start = self.valid_date - datetime.timedelta(hours=wanted[1])
            end = self.valid_date - datetime.timedelta(hours=wanted[0])

            found = self.search_intervals(start, end)
            if not found:
                print(f"❌❌❌ Cannot find accumulation for {start} {end}")
                self.search_intervals(start, end, debug=True)
                raise ValueError(f"Cannot find accumulation for {start} {end}")

            # choosing most recent forecast
            found = sorted(found, key=lambda x: x.base_datetime, reverse=True)
            chosen = found[0]

            if len(found) > 1:
                print(f"  Found more than one interval for {start} {end}")
                for f in found:
                    print(f"    {f}")
                print(f"    Chosing {chosen}, base_datetime: {chosen.base_datetime}")

            chosen.sign = 1

            lst.append(chosen)
        return lst


class EaOperIntervalsCollection(EraIntervalsCollection):
    def available_steps(self, start, end) -> dict:
        """Return the IntervalsCollection time steps available to build/search an available interval

        Parameters:
        ----------
            start (datetime.datetime): step (=leadtime) from the forecast where interval begins
            end (datetime.datetime): step (=leadtime) from the forecast where interval ends

        Return:
        -------
            _ (dict[List[int]]) :  dictionary listing the available steps between start and end for each base

        """
        return {
            6: [[i, i + 1] for i in range(0, 18, 1)],
            18: [[i, i + 1] for i in range(0, 18, 1)],
        }


class L5OperIntervalsCollection(EraIntervalsCollection):
    def available_steps(self, start, end) -> dict:
        """Return the IntervalsCollection time steps available to build/search an available interval

        Parameters:
        ----------
            start (datetime.datetime): step (=leadtime) from the forecast where interval begins
            end (datetime.datetime): step (=leadtime) from the forecast where interval ends

        Return:
        -------
            _ (dict[List[int]]) :  dictionary listing the available steps between start and end for each base

        """
        print("❌❌❌ untested")
        x = 24  # need to check if 24 is the right value
        return {
            0: [[i, i + 1] for i in range(0, x, 1)],
        }


class EaEndaIntervalsCollection(EraIntervalsCollection):
    def available_steps(self, start, end) -> dict:
        """Return the IntervalsCollection time steps available to build/search an available interval

        Parameters:
        ----------
            start (datetime.datetime): step (=leadtime) from the forecast where interval begins
            end (datetime.datetime): step (=leadtime) from the forecast where interval ends

        Return:
        -------
            _ (dict[List[int]]) :  dictionary listing the available steps between start and end for each base

        """
        print("❌❌❌ untested")
        return {
            6: [[i, i + 3] for i in range(0, 18, 1)],
            18: [[i, i + 3] for i in range(0, 18, 1)],
        }


class RrOperIntervalsCollection(IntervalsCollection):

    forecast_period: datetime.timedelta = datetime.timedelta(hours=3)

    def available_steps(self) -> dict:
        """Return the IntervalsCollection time steps available to build/search an available interval

        Parameters:
        ----------
            start (datetime.datetime): step (=leadtime) from the forecast where interval begins
            end (datetime.datetime): step (=leadtime) from the forecast where interval ends

        Return:
        -------
            _ (dict[List[int]]) :  dictionary listing the available steps between start and end for each base

        """
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

    def search_interval(self, current: datetime.datetime) -> Interval:

        steps = self.available_steps()
        current_hour = current.hour

        if current_hour >= int(self.forecast_period.seconds // 3600):
            start_guess = max([sk for sk in steps if sk < current_hour])
        else:
            start_guess = max(list(steps.keys()))
        assert [0, (current_hour - start_guess) % 24] in steps[start_guess], f"No available forecast for {current}"

        delta = datetime.timedelta(hours=(current_hour - start_guess) % 24)
        assert delta <= self.forecast_period, f"{current} is too far from a forecast start"
        start = current - delta

        return Interval(start, current, start)

    def build_intervals(self) -> list[Interval]:
        """Build the list of intervals to accumulate the data on the IntervalsCollection

        available steps --> checking the closest base datetime for each hour
        difference should be implemented
        """

        start = self.valid_date - self.accumulation_period
        current = self.valid_date
        lst = []
        while current > start:
            chosen = self.search_interval(current)
            current = current - (chosen.end_datetime - chosen.start_datetime)
            print(f"start : {start}, current: {current}, {chosen}")
            # avoid infinite loop
            assert current < chosen.end_datetime
            chosen.sign = 1
            # if start matches a forecast start, no additional interval needed
            lst.append(chosen)
            print(chosen)

        # if start does not match a forecast start, one additional interval needed
        if current < start:
            print(f"additional step start : {start}, current: {current}")
            assert start - current < self.forecast_period, f"{current} selects a forecast too far in the past"
            assert current.hour in self.available_steps()
            chosen = Interval(current, start, current)
            # must substract the accumulated value from forecast start
            chosen.sign = -1
            lst.append(chosen)
            print(chosen)
        return lst


class OdEldaIntervalsCollection(EraIntervalsCollection):
    def available_steps(self, start, end) -> dict:
        """Return the IntervalsCollection time steps available to build/search an available interval

        Parameters:
        ----------
            start (datetime.datetime): step (=leadtime) from the forecast where interval begins
            end (datetime.datetime): step (=leadtime) from the forecast where interval ends

        Return:
        -------
            _ (dict[List[int]]) :  dictionary listing the available steps between start and end for each base

        """
        print("❌❌❌ untested")
        x = 24  # need to check if 24 is the right value
        return {
            6: [[i, i + 1] for i in range(0, x, 1)],
            18: [[i, i + 1] for i in range(0, x, 1)],
        }


class DiffIntervalsCollection(IntervalsCollection):
    pass


class OdOperIntervalsCollection(DiffIntervalsCollection):
    def available_steps(self, start, end) -> dict:
        raise NotImplementedError("need to implement diff and _scda patch (cf accumulation_utils.utils)")


class OdEnfoIntervalsCollection(DiffIntervalsCollection):
    def available_steps(self, start, end) -> dict:
        raise NotImplementedError("need to implement diff")


def find_IntervalsCollection_class(request: dict[str, Any]) -> type[IntervalsCollection]:
    """Find the appropriate IntervalsCollection class given the recipe's parameters

    Parameters:
    ----------
    request: dict[str, Any]
        The recipes request parameters

    """
    try:
        return {
            ("ea", "oper"): EaOperIntervalsCollection,
            ("ea", None): EaOperIntervalsCollection,
            ("ea", "enda"): EaEndaIntervalsCollection,
            ("rr", "oper"): RrOperIntervalsCollection,
            ("l5", "oper"): L5OperIntervalsCollection,
            ("od", "oper"): OdOperIntervalsCollection,
            ("od", "enfo"): OdEnfoIntervalsCollection,
            ("od", "elda"): OdEldaIntervalsCollection,
        }[request.get("class", None), request.get("stream", None)]

    except KeyError:
        return DefaultIntervalsCollection
