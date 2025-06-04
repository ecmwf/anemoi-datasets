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
import warnings
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import earthkit.data as ekd
import numpy as np
from earthkit.data.core.temporary import temp_file
from earthkit.data.readers.grib.output import new_grib_output
from numpy.typing import NDArray

from anemoi.datasets.create.utils import to_datetime_list

from .legacy import legacy_source
from .mars import mars

LOG = logging.getLogger(__name__)


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


class Accumulation:
    """Class to handle data accumulation for a specific parameter, date, time, and member."""

    buggy_steps: bool = False

    def __init__(
        self,
        out: Any,
        /,
        param: str,
        date: int,
        time: int,
        number: int,
        step: List[int],
        frequency: int,
        accumulations_reset_frequency: Optional[int] = None,
        user_date: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialises an Accumulation instance.

        Parameters
        ----------
        out : Any
            Output object for writing data.
        param : str
            Parameter name.
        date : int
            Date of the accumulation.
        time : int
            Time of the accumulation.
        number : int
            Member number.
        step : List[int]
            List of steps.
        frequency : int
            Frequency of accumulation.
        accumulations_reset_frequency : Optional[int], optional
            Frequency at which accumulations reset. Defaults to None.
        user_date : Optional[str], optional
            User-defined date. Defaults to None.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.out = out
        self.param = param
        self.date = date
        self.time = time
        self.steps = step
        self.number = number
        self.values: Optional[NDArray[None]] = None
        self.seen = set()
        self.startStep: Optional[int] = None
        self.endStep: Optional[int] = None
        self.done = False
        self.frequency = frequency
        self.accumulations_reset_frequency = accumulations_reset_frequency
        self._check = None
        self.user_date = user_date

    @property
    def key(self) -> Tuple[str, int, int, List[int], int]:
        """Returns the key for the accumulation."""
        return (self.param, self.date, self.time, self.steps, self.number)

    def check(self, field: Any) -> None:
        """Checks the field metadata against the accumulation parameters.

        Parameters
        ----------
        field : Any
            The field to check.
        """
        if self._check is None:
            self._check = field.metadata(namespace="mars")

            assert self.param == field.metadata("param"), (
                self.param,
                field.metadata("param"),
            )
            assert self.date == field.metadata("date"), (
                self.date,
                field.metadata("date"),
            )
            assert self.time == field.metadata("time"), (
                self.time,
                field.metadata("time"),
            )
            assert self.number == _member(field), (self.number, _member(field))

            return

        mars = field.metadata(namespace="mars")
        keys1 = sorted(self._check.keys())
        keys2 = sorted(mars.keys())

        assert keys1 == keys2, (keys1, keys2)

        for k in keys1:
            if k not in ("step",):
                assert self._check[k] == mars[k], (k, self._check[k], mars[k])

    def write(self, template: Any) -> None:
        """Writes the accumulated values to the output.

        Parameters
        ----------
        template : Any
            Template for writing the output.
        """
        assert self.startStep != self.endStep, (self.startStep, self.endStep)
        if np.all(self.values < 0):
            LOG.warning(
                f"Negative values when computing accumutation for {self.param} ({self.date} {self.time}): min={np.amin(self.values)} max={np.amax(self.values)}"
            )

        # In GRIB1, is the step is greater that 254 (one byte), we cannot use a range, because both P1 and P2 values
        # are used to store the end step

        edition = template.metadata("edition")

        if edition == 1 and self.endStep > 254:
            self.out.write(
                self.values,
                template=template,
                stepType="instant",
                step=self.endStep,
                check_nans=True,
            )
        else:
            self.out.write(
                self.values,
                template=template,
                stepType="accum",
                startStep=self.startStep,
                endStep=self.endStep,
                check_nans=True,
            )
        self.values = None
        self.done = True

    def add(self, field: Any, values: NDArray[Any]) -> None:
        """Adds values to the accumulation.

        Parameters
        ----------
        field : Any
            The field containing the values.
        values : NDArray[Any]
            The values to add.
        """
        self.check(field)

        step = field.metadata("step")
        if step not in self.steps:
            return

        if not np.all(values >= 0):
            warnings.warn(f"Negative values for {field}: {np.nanmin(values)} {np.nanmax(values)}")

        assert not self.done, (self.key, step)
        assert step not in self.seen, (self.key, step)

        startStep = field.metadata("startStep")
        endStep = field.metadata("endStep")

        if startStep == endStep:
            startStep, endStep = self.adjust_steps(startStep, endStep)

        assert step == endStep, (startStep, endStep, step)

        self.compute(values, startStep, endStep)

        self.seen.add(step)

        if len(self.seen) == len(self.steps):
            self.write(template=field)

    @classmethod
    def mars_date_time_steps(
        cls,
        *,
        dates: List[datetime.datetime],
        step1: int,
        step2: int,
        frequency: Optional[int],
        base_times: List[int],
        adjust_step: bool,
        accumulations_reset_frequency: Optional[int],
        user_date: Optional[str],
    ) -> Generator[Tuple[int, int, Tuple[int, ...]], None, None]:
        """Generates MARS date-time steps.

        Parameters
        ----------
        dates : List[datetime.datetime]
            List of dates.
        step1 : int
            First step.
        step2 : int
            Second step.
        frequency : Optional[int]
            Frequency of accumulation.
        base_times : List[int]
            List of base times.
        adjust_step : bool
            Whether to adjust the step.
        accumulations_reset_frequency : Optional[int], optional
            Frequency at which accumulations reset. Defaults to None.
        user_date : Optional[str], optional
            User-defined date. Defaults to None.

        Returns
        -------
        Generator[Tuple[int, int, Tuple[int, ...]], None, None]
            A generator of MARS date-time steps.
        """
        # assert step1 > 0, (step1, step2, frequency)

        for valid_date in dates:
            add_step = 0
            base_date = valid_date - datetime.timedelta(hours=step2)
            if user_date is not None:
                assert user_date == "????-??-01", user_date
                new_base_date = base_date.replace(day=1)
                assert new_base_date <= base_date, (new_base_date, base_date)
                add_step = int((base_date - new_base_date).total_seconds() // 3600)

                base_date = new_base_date

            if base_date.hour not in base_times:
                if not adjust_step:
                    raise ValueError(
                        f"Cannot find a base time in {base_times} that validates on {valid_date.isoformat()} for step={step2}"
                    )

                while base_date.hour not in base_times:
                    # print(f'{base_date=}, {base_times=}, {add_step=} {frequency=}')
                    base_date -= datetime.timedelta(hours=1)
                    add_step += 1

            yield cls._mars_date_time_step(
                base_date=base_date,
                step1=step1,
                step2=step2,
                add_step=add_step,
                frequency=frequency,
                accumulations_reset_frequency=accumulations_reset_frequency,
                user_date=user_date,
                requested_date=valid_date,
            )

    def compute(self, values: NDArray[Any], startStep: int, endStep: int) -> None:
        """Computes the accumulation.

        Parameters
        ----------
        values : NDArray[Any]
            The values to accumulate.
        startStep : int
            The start step.
        endStep : int
            The end step.
        """
        pass

    @classmethod
    def _mars_date_time_step(
        cls,
        *,
        base_date: datetime.datetime,
        step1: int,
        step2: int,
        add_step: int,
        frequency: Optional[int],
        accumulations_reset_frequency: Optional[int],
        user_date: Optional[str],
        requested_date: Optional[datetime.datetime] = None,
    ) -> Tuple[int, int, Tuple[int, ...]]:
        """Generates a MARS date-time step.

        Parameters
        ----------
        base_date : datetime.datetime
            The base date.
        step1 : int
            First step.
        step2 : int
            Second step.
        add_step : int
            Additional step.
        frequency : Optional[int]
            Frequency of accumulation.
        accumulations_reset_frequency : Optional[int], optional
            Frequency at which accumulations reset. Defaults to None.
        user_date : Optional[str], optional
            User-defined date. Defaults to None.
        requested_date : Optional[datetime.datetime], optional
            Requested date. Defaults to None.

        Returns
        -------
        Tuple[int, int, Tuple[int, ...]]
            A tuple representing the MARS date-time step.
        """
        pass


class AccumulationFromStart(Accumulation):
    """Class to handle data accumulation from the start of the forecast."""

    def adjust_steps(self, startStep: int, endStep: int) -> Tuple[int, int]:
        """Adjusts the start and end steps.

        Parameters
        ----------
        startStep : int
            The start step.
        endStep : int
            The end step.

        Returns
        -------
        Tuple[int, int]
            The adjusted start and end steps.
        """
        assert endStep == startStep
        return (0, endStep)

    def compute(self, values: NDArray[Any], startStep: int, endStep: int) -> None:
        """Computes the accumulation from the start.

        Parameters
        ----------
        values : np.ndarray
            The values to accumulate.
        startStep : int
            The start step.
        endStep : int
            The end step.
        """
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
    def _mars_date_time_step(
        cls,
        *,
        base_date: datetime.datetime,
        step1: int,
        step2: int,
        add_step: int,
        frequency: Optional[int],
        accumulations_reset_frequency: Optional[int],
        user_date: Optional[str],
        requested_date: Optional[datetime.datetime] = None,
    ) -> Tuple[int, int, Tuple[int, ...]]:
        """Generates a MARS date-time step.

        Parameters
        ----------
        base_date : datetime.datetime
            The base date.
        step1 : int
            First step.
        step2 : int
            Second step.
        add_step : int
            Additional step.
        frequency : Optional[int]
            Frequency of accumulation.
        accumulations_reset_frequency : Optional[int], optional
            Frequency at which accumulations reset. Defaults to None.
        user_date : Optional[str], optional
            User-defined date. Defaults to None.
        requested_date : Optional[datetime.datetime], optional
            Requested date. Defaults to None.

        Returns
        -------
        Tuple[int, int, Tuple[int, ...]]
            A tuple representing the MARS date-time step.
        """
        assert user_date is None, user_date
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
    """Class to handle data accumulation from the last step of the forecast."""

    def compute(self, values: NDArray[Any], startStep: int, endStep: int) -> None:
        """Computes the accumulation from the last step.

        Parameters
        ----------
        values : np.ndarray
            The values to accumulate.
        startStep : int
            The start step.
        endStep : int
            The end step.
        """
        assert endStep - startStep == self.frequency, (
            startStep,
            endStep,
            self.frequency,
        )

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
    def _mars_date_time_step(
        cls,
        *,
        base_date: datetime.datetime,
        step1: int,
        step2: int,
        add_step: int,
        frequency: int,
        accumulations_reset_frequency: Optional[int],
        user_date: Optional[str] = None,
        requested_date: Optional[datetime.datetime] = None,
    ) -> Tuple[int, int, Tuple[int, ...]]:
        """Generates a MARS date-time step.

        Parameters
        ----------
        base_date : datetime.datetime
            The base date.
        step1 : int
            First step.
        step2 : int
            Second step.
        add_step : int
            Additional step.
        frequency : int
            Frequency of accumulation.
        accumulations_reset_frequency : Optional[int], optional
            Frequency at which accumulations reset. Defaults to None.
        user_date : Optional[str], optional
            User-defined date. Defaults to None.
        requested_date : Optional[datetime.datetime], optional
            Requested date. Defaults to None.

        Returns
        -------
        Tuple[int, int, Tuple[int, ...]]
            A tuple representing the MARS date-time step.
        """

        assert user_date is None, user_date

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


class AccumulationFromLastReset(Accumulation):
    """Class to handle data accumulation from the last step of the forecast."""

    def adjust_steps(self, startStep: int, endStep: int) -> Tuple[int, int]:
        """Adjusts the start and end steps.

        Parameters
        ----------
        startStep : int
            The start step.
        endStep : int
            The end step.

        Returns
        -------
        Tuple[int, int]
            The adjusted start and end steps.
        """
        return self.__class__._adjust_steps(startStep, endStep, self.frequency, self.accumulations_reset_frequency)

    @classmethod
    def _adjust_steps(
        self, startStep: int, endStep: int, frequency: int, accumulations_reset_frequency: int
    ) -> Tuple[int, int]:
        """Adjusts the start and end steps.

        Parameters
        ----------
        startStep : int
            The start step.
        endStep : int
            The end step.
        frequency : int
            Frequency of accumulation.
        accumulations_reset_frequency : int
            Frequency at which accumulations reset.

        Returns
        -------
        Tuple[int, int]
            The adjusted start and end steps.
        """

        assert frequency == 1, (frequency, startStep, endStep)
        assert endStep - startStep <= accumulations_reset_frequency, (startStep, endStep)

        return ((startStep // accumulations_reset_frequency) * accumulations_reset_frequency, endStep)

    @classmethod
    def _steps(
        cls,
        valid_date: datetime.datetime,
        base_date: datetime.datetime,
        frequency: int,
        accumulations_reset_frequency: int,
    ) -> Tuple[int, int]:
        """Calculates the steps for accumulation.

        Parameters
        ----------
        valid_date : datetime.datetime
            The valid date.
        base_date : datetime.datetime
            The base date.
        frequency : int
            Frequency of accumulation.
        accumulations_reset_frequency : int
            Frequency at which accumulations reset.

        Returns
        -------
        Tuple[int, int]
            A tuple representing the steps for accumulation.
        """

        assert base_date.day == 1, (base_date, valid_date)

        step = (valid_date - base_date).total_seconds()
        assert int(step) == step, (valid_date, base_date, step)
        assert int(step) % 3600 == 0, (valid_date, base_date, step)
        step = int(step // 3600)
        start, end = cls._adjust_steps(step, step, frequency, accumulations_reset_frequency)

        return start + frequency, end

    def compute(self, values: NDArray[Any], startStep: int, endStep: int) -> None:
        """Computes the accumulation from the last step.

        Parameters
        ----------
        values : NDArray[Any]
            The values to accumulate.
        startStep : int
            The start step.
        endStep : int
            The end step.
        """

        assert self.frequency == 1

        assert startStep % self.accumulations_reset_frequency == 0, (
            startStep,
            endStep,
            self.accumulations_reset_frequency,
        )

        if self.values is None:

            self.values = np.copy(values)
            self.startStep = startStep
            self.endStep = endStep

            if len(self.steps) == 1:
                assert self.startStep == self.endStep - self.frequency, (self.startStep, self.endStep)

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

            assert self.endStep - self.startStep <= self.accumulations_reset_frequency, (self.startStep, startStep)

    @classmethod
    def _mars_date_time_step(
        cls,
        *,
        base_date: datetime.datetime,
        step1: int,
        step2: int,
        add_step: int,
        frequency: int,
        accumulations_reset_frequency: Optional[int],
        user_date: Optional[str],
        requested_date: Optional[datetime.datetime] = None,
    ) -> Tuple[int, int, Tuple[int, ...]]:
        """Generates a MARS date-time step.

        Parameters
        ----------
        base_date : datetime.datetime
            The base date.
        step1 : int
            First step.
        step2 : int
            Second step.
        add_step : int
            Additional step.
        frequency : int
            Frequency of accumulation.
        accumulations_reset_frequency : Optional[int]
            Frequency at which accumulations reset.
        user_date : Optional[str]
            User-defined date.
        requested_date : Optional[datetime.datetime], optional
            Requested date. Defaults to None.

        Returns
        -------
        Tuple[int, int, Tuple[int, ...]]
            A tuple representing the MARS date-time step.
        """
        # assert frequency > 0, frequency
        # assert step1 > 0, (step1, step2, frequency, add_step, base_date)

        step1 += add_step
        step2 += add_step

        assert step2 - step1 == frequency, (step1, step2, frequency)

        adjust_step1 = cls._adjust_steps(step1, step1, frequency, accumulations_reset_frequency)
        adjust_step2 = cls._adjust_steps(step2, step2, frequency, accumulations_reset_frequency)

        if adjust_step1[1] % accumulations_reset_frequency == 0:
            # First step of a new accumulation
            steps = (adjust_step2[1],)
        else:
            steps = (adjust_step1[1], adjust_step2[1])

        return (
            base_date.year * 10000 + base_date.month * 100 + base_date.day,
            base_date.hour * 100 + base_date.minute,
            tuple(steps),
        )


def _identity(x: Any) -> Any:
    """Identity function that returns the input as is.

    Parameters
    ----------
    x : Any
        Input value.

    Returns
    -------
    Any
        The input value.
    """
    return x


def _compute_accumulations(
    context: Any,
    dates: List[datetime.datetime],
    request: Dict[str, Any],
    user_accumulation_period: Union[int, Tuple[int, int]] = 6,
    data_accumulation_period: Optional[int] = None,
    accumulations_reset_frequency: Optional[int] = None,
    user_date: Optional[str] = None,
    patch: Any = _identity,
    base_times: Optional[List[int]] = None,
    use_cdsapi_dataset: Optional[str] = None,
) -> Any:
    """Computes accumulations based on the provided parameters.

    Parameters
    ----------
    context : Any
        Context for the computation.
    dates : List[datetime.datetime]
        List of dates.
    request : Dict[str, Any]
        Request parameters.
    user_accumulation_period : Union[int, Tuple[int, int]], optional
        User-defined accumulation period. Defaults to 6.
    data_accumulation_period : Optional[int], optional
        Data accumulation period. Defaults to None.
    accumulations_reset_frequency : Optional[int], optional
        Frequency at which accumulations reset. Defaults to None.
    user_date : Optional[str], optional
        User-defined date. Defaults to None.
    patch : Any, optional
        Patch function. Defaults to _identity.
    base_times : Optional[List[int]], optional
        List of base times. Defaults to None.
    use_cdsapi_dataset : Optional[str], optional
        CDSAPI dataset to use. Defaults to None.

    Returns
    -------
    Any
        The computed accumulations.
    """
    adjust_step = isinstance(user_accumulation_period, int)

    if not isinstance(user_accumulation_period, (list, tuple)):
        user_accumulation_period = (0, user_accumulation_period)

    assert len(user_accumulation_period) == 2, user_accumulation_period
    step1, step2 = user_accumulation_period
    assert step1 < step2, user_accumulation_period

    if data_accumulation_period is None:
        data_accumulation_period = user_accumulation_period[1] - user_accumulation_period[0]

    if base_times is None:
        if "time" in request:
            time = request.pop("time")
            if time > 100:
                time = time // 100
            base_times = [time]
        else:
            base_times = [0, 6, 12, 18]

    base_times = [t // 100 if t > 100 else t for t in base_times]

    if accumulations_reset_frequency is not None:
        AccumulationClass = AccumulationFromLastReset
    else:
        AccumulationClass = AccumulationFromStart if data_accumulation_period in (0, None) else AccumulationFromLastStep

    mars_date_time_steps = AccumulationClass.mars_date_time_steps(
        dates=dates,
        step1=step1,
        step2=step2,
        frequency=data_accumulation_period,
        base_times=base_times,
        adjust_step=adjust_step,
        accumulations_reset_frequency=accumulations_reset_frequency,
        user_date=user_date,
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
                r = dict(request, param=p, date=date, time=time, step=sorted(steps), number=n)

                requests.append(patch(r))

    ds = mars(
        context, dates, *requests, request_already_using_valid_datetime=True, use_cdsapi_dataset=use_cdsapi_dataset
    )

    accumulations = {}
    for a in [
        AccumulationClass(out, frequency=frequency, accumulations_reset_frequency=accumulations_reset_frequency, **r)
        for r in requests
    ]:
        for s in a.steps:
            key = (a.param, a.date, a.time, s, a.number)
            accumulations.setdefault(key, []).append(a)

    for field in ds:
        key = (
            field.metadata("param"),
            field.metadata("date"),
            field.metadata("time"),
            field.metadata("step"),
            _member(field),
        )
        values = field.values  # optimisation
        if key not in accumulations:
            raise ValueError(f"Key not found: {key}. Is it an accumulation field?")

        for a in accumulations[key]:
            a.add(field, values)

    for acc in accumulations.values():
        for a in acc:
            assert a.done, (a.key, a.seen, a.steps)

    out.close()

    ds = ekd.from_source("file", path)

    assert len(ds) / len(param) / len(number) == len(dates), (
        len(ds),
        len(param),
        len(dates),
    )
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
def accumulations(
    context: Any, dates: List[datetime.datetime], use_cdsapi_dataset: Optional[str] = None, **request: Any
) -> Any:
    """Computes accumulations based on the provided context, dates, and request parameters.

    Parameters
    ----------
    context : Any
        Context for the computation.
    dates : List[datetime.datetime]
        List of dates.
    use_cdsapi_dataset : Optional[str], optional
        CDSAPI dataset to use. Defaults to None.
    **request : Any
        Additional request parameters.

    Returns
    -------
    Any
        The computed accumulations.
    """

    if (
        request.get("class") == "ea"
        and request.get("stream", "oper") == "oper"
        and request.get("accumulation_period") == 24
    ):
        from .accumulations2 import accumulations as accumulations2

        LOG.warning(
            "🧪️ Experimental features: Using accumulations2, because class=ea stream=oper and accumulation_period=24"
        )
        return accumulations2(context, dates, **request)

    _to_list(request["param"])
    class_ = request.get("class", "od")
    stream = request.get("stream", "oper")

    user_accumulation_period = request.pop("accumulation_period", 6)
    accumulations_reset_frequency = request.pop("accumulations_reset_frequency", None)
    user_date = request.pop("date", None)

    # If `data_accumulation_period` is not set, this means that the accumulations are from the start
    # of the forecast.

    KWARGS = {
        ("od", "oper"): dict(patch=_scda),
        ("od", "elda"): dict(base_times=(6, 18)),
        ("od", "enfo"): dict(base_times=(0, 6, 12, 18)),
        ("ea", "oper"): dict(data_accumulation_period=1, base_times=(6, 18)),
        ("ea", "enda"): dict(data_accumulation_period=3, base_times=(6, 18)),
        ("rr", "oper"): dict(base_times=(0, 3, 6, 9, 12, 15, 18, 21)),
        ("l5", "oper"): dict(data_accumulation_period=1, base_times=(0,)),
    }

    kwargs = KWARGS.get((class_, stream), {})

    context.trace("🌧️", f"accumulations {request} {user_accumulation_period} {kwargs}")

    return _compute_accumulations(
        context,
        dates,
        request,
        user_accumulation_period=user_accumulation_period,
        accumulations_reset_frequency=accumulations_reset_frequency,
        use_cdsapi_dataset=use_cdsapi_dataset,
        user_date=user_date,
        **kwargs,
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
    """
    )
    dates = yaml.safe_load("[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]")
    dates = to_datetime_list(dates)

    for f in accumulations(None, dates, **config):
        print(f, f.to_numpy().mean())
