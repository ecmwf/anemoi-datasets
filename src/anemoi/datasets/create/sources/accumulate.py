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
from typing import Any

import earthkit.data
import numpy as np
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.core.temporary import temp_file
from earthkit.data.readers.grib.output import new_grib_output
from numpy.typing import NDArray

from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.create.sources.accumulate_utils.interval_generators import interval_generator_factory

from .accumulate_utils.covering_intervals import SignedInterval
from .legacy import LegacySource

LOG = logging.getLogger(__name__)

DEBUG = True
trace = print if DEBUG else lambda *args, **kwargs: None


def _adjust_request_to_interval(interval: Any, request: list[dict]) -> tuple[Any]:
    # TODO:
    # for od-oper: need to do this adjustment, should be in mars source itself?
    # Modifies the request stream based on the time (so, not here).
    # if request["time"] in (6, 18, 600, 1800):
    #    request["stream"] = "scda"
    # else:
    #    request["stream"] = "oper"
    r = request.copy()
    if interval.base is None:
        # for some sources, we may not have a base time (grib-index)
        step = int((interval.end - interval.start).total_seconds() / 3600)
        r["step"] = step
        return interval.max, request, step
    else:
        step = int((interval.end - interval.base).total_seconds() / 3600)
        r["date"] = interval.base.strftime("%Y%m%d")
        r["time"] = interval.base.strftime("%H%M")
        r["step"] = step
        return interval.max, r, step


class Accumulator:
    values: NDArray | None = None
    locked: bool = False

    def __init__(self, valid_date: datetime.datetime, period: datetime.timedelta, key: dict[str, Any], coverage):
        # The accumulator only accumulates fields and does not know about the rest
        # Accumulator object for a given param/member/valid_date

        self.valid_date = valid_date
        self.period = period
        self.key = key

        self.coverage = coverage

        self.todo = [v for v in coverage]
        self.done = []

        self.values = None  # will hold accumulated values array

    def is_complete(self, **kwargs) -> bool:
        """Check whether the accumulation is complete (all intervals have been processed)"""
        return not self.todo

    def compute(self, field: Any, values: NDArray, interval) -> None:
        """Perform accumulation with the values array on this interval and record the operation.
        Note: values have been extracted from field before the call to `compute`,
        so values are read from field only once.

        Parameters:
        ----------
        field: Any
            An earthkit-data-like field
        values: NDArray
            Values from the field, will be added to the held values array

        Return
        ------
        None
        """

        def match_interval(interval: SignedInterval, lst: list[SignedInterval]) -> bool:
            for i in lst:
                if i.start == interval.start or i.end == interval.end and i.base == interval.base:
                    print(f"✅ {interval} == {i}")
                    return i
                if i.start == interval.start and i.end == interval.end and i.base is None:
                    print(f"✅ {interval} ~= {i}")
                    return i
                print(f"❌ {interval} != {i}")
            return None

        matching = match_interval(interval, self.todo)

        if not matching:
            # interval not needed for this accumulator
            # this happens when multiple accumulators have the same key but different valid_date
            return False

        def raise_error(msg):
            LOG.error(f"Accumulator {self.__repr__(verbose=True)} state:")
            LOG.error(f"Received interval: {interval}")
            LOG.error(f"Matching interval: {matching}")
            raise ValueError(msg)

        if matching in self.done:
            # this should not happen normally
            raise_error(f"SignedInterval {matching} already done for accumulator")

        if self.locked:
            raise_error(f"Accumulator already used, cannot process interval {interval}")

        assert isinstance(values, np.ndarray), type(values)

        # actual accumulation computation
        # negative accumulation if interval is reversed
        # copy is mandatory since value is shared between accumulators
        local_values = interval.sign * values.copy()
        if self.values is None:
            self.values = local_values
        else:
            self.values += local_values

        self.todo.remove(matching)
        self.done.append(matching)
        return True

    def write_to_output(self, output, template) -> None:
        assert self.is_complete(), (self.todo, self.done, self)
        assert not self.locked  # prevent double writing

        # negative values may be an anomaly (e.g precipitation), but this is user's choice
        if np.any(self.values < 0):
            LOG.warning(
                f"Negative values when computing accumutation for {self}): min={np.amin(self.values)} max={np.amax(self.values)}"
            )
        write_accumulated_field_with_valid_time(
            template=template,
            values=self.values,
            valid_date=self.valid_date,
            period=self.period,
            output=output,
        )
        # lock the accumulator to prevent further use
        self.locked = True

    def __repr__(self, verbose: bool = False) -> str:
        key = ", ".join(f"{k}={v}" for k, v in self.key)
        period = frequency_to_string(self.period)
        default = f"{self.__class__.__name__}(valid_date={self.valid_date}, {period}, key={{ {key} }})"
        if verbose:
            extra = []
            if self.locked:
                extra.append("(locked)")
            for i in self.done:
                extra.append(f"    done: {i}")
            for i in self.todo:
                extra.append(f"    todo: {i}")
            default += "\n" + "\n".join(extra)
        return default


def write_accumulated_field_with_valid_time(
    template, values, valid_date: datetime.datetime, period: datetime.timedelta, output
) -> Any:
    MISSING_VALUE = 1e-38
    assert np.all(values != MISSING_VALUE)

    date = (valid_date - period).strftime("%Y%m%d")
    time = (valid_date - period).strftime("%H%M")
    endStep = period

    hours = endStep.total_seconds() / 3600
    if not hours.is_integer():
        raise ValueError(f"Accumulation period must be integer hours, got {hours}")
    hours = int(hours)

    if template.metadata("edition") == 1 and (hours > 254 or not hours.is_integer()):
        # this is a special case for GRIB edition 1 which only supports integer hours up to 254
        assert hours.is_integer(), f"edition 1 accumulation period must be integer hours, got {hours}"
        assert hours <= 254, f"edition 1 accumulation period must be <=254 hours, got {hours}"
        output.write(
            values,
            template=template,
            date=int(date),
            time=int(time),
            stepType="instant",
            step=hours,
            check_nans=True,
            missing_value=MISSING_VALUE,
        )
    else:
        # this is the normal case for GRIB edition 2. And with edition 1 when hours are integer and <=254
        output.write(
            values,
            template=template,
            date=int(date),
            time=int(time),
            stepType="accum",
            startStep=0,
            endStep=hours,
            check_nans=True,
            missing_value=MISSING_VALUE,
        )


def field_to_interval(field):
    date_str = str(field.metadata("date")).zfill(8)
    time_str = str(field.metadata("time")).zfill(4)
    base_datetime = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")

    endStep = field.metadata("endStep")
    startStep = field.metadata("startStep")
    typeStep = field.metadata("stepType")

    # if endStep == 0 and startStep > 0:
    #   startStep, endStep = endStep, startStep

    if startStep == endStep:
        startStep = 0
        assert typeStep == "instant", "If startStep == endStep, stepType must be 'instant'"
    assert startStep < endStep, (startStep, endStep)

    start_step = datetime.timedelta(hours=startStep)
    end_step = datetime.timedelta(hours=endStep)

    trace(f"    field: {startStep=}, {endStep=}")

    interval = SignedInterval(
        start=base_datetime + start_step,
        end=base_datetime + end_step,
        base=base_datetime,
    )

    date_str = str(field.metadata("validityDate")).zfill(8)
    time_str = str(field.metadata("validityTime")).zfill(4)
    valid_date = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    assert valid_date == interval.max, (valid_date, interval)

    trace(f"    field interval: {interval}")

    return interval


def _compute_accumulations(
    context: Any,
    dates: list[datetime.datetime],
    period: datetime.timedelta,
    source: Any,
    available: dict[str, Any] | None = None,
    **kwargs,
) -> Any:
    """Concrete accumulation logic.

    - identify the needed intervals for each date/parameter/member defined in recipe
    - fetch the source data via a database (mars or grib-index)
    - create Accumulator objects and fill them will accumulated values from source data
    - return the datasource with accumulated values

    Parameters:
    ----------
    context: Any,
        The dataset building context (will be updated with trace of accumulation)
    dates: list[datetime.datetime]
        The list of valid dates on which to perform accumulations.
    source: Any,
    period: datetime.timedelta,
        The interval over which to accumulate (user-defined)
    available: Any, optional
        A description of the available periods in the data source. See documentation.

    Return
    ------
    The accumulated datasource for all dates, parameters, members.

    """

    LOG.debug("💬 source for accumulations: %s", source)

    assert isinstance(source, dict)
    assert len(source) == 1, f"Source must have exactly one key, got {list(source.keys())}"
    source_name, _ = next(iter(source.items()))
    if source_name == "mars":
        if "type" not in source[source_name]:
            source[source_name]["type"] = "fc"
            LOG.warning("Assuming 'type: fc' for mars source as it was not specified in the recipe")
        if "levtype" not in source[source_name]:
            source[source_name]["levtype"] = "sfc"
            LOG.warning("Assuming 'levtype: sfc' for mars source as it was not specified in the recipe")

    interval_generator = interval_generator_factory(available)
    if not isinstance(period, datetime.timedelta):
        period = frequency_to_timedelta(period)

    coverages = {}
    for d in dates:
        if not isinstance(d, datetime.datetime):
            raise TypeError("valid_date must be a datetime.datetime instance")
        coverages[d] = interval_generator.covering_intervals(d - period, d)
        trace(f"  Found covering intervals: for {d - period} to {d}:")
        for c in coverages[d]:
            trace(f"    {c}")
    # this piece of code is calling for a class Intervals()
    # dates = Intervals(dates, ...)
    dates.date_to_intervals = coverages
    dates._adjust_request_to_interval = _adjust_request_to_interval

    def _intervals():
        for d in dates:
            for interval in coverages[d]:
                yield d, interval

    dates.intervals = _intervals()

    h = hashlib.md5(json.dumps((str(period), source), sort_keys=True).encode()).hexdigest()
    print(source)
    source_object = context.create_source(source, "data_sources", h)

    # need a temporary file to store the accumulated fields for now, because earthkit-data
    # does not completely support in-memory fieldlists yet (metadata consistency is not fully ensured)
    tmp = temp_file()
    path = tmp.path
    output = new_grib_output(path)

    accumulators = {}
    for field in source_object(context, dates):
        # for each field provided by the catalogue, find which accumulators need it and perform accumulation

        values = field.values.copy()
        # would values = field.values() be enough ?

        key = field.metadata(namespace="mars")
        key = {k: v for k, v in key.items() if k not in ["date", "time", "step"]}
        key = tuple(sorted(key.items()))
        print("---")
        print(f"\033[93m FIELD {field}, key: {key}\033[0m")

        interval = field_to_interval(field)
        print(f"    -> interval: {interval}")

        valid_date = interval.max

        field_used = False
        for date in dates:
            # build accumulator if it does not exist yet
            print(f"  \033[94mChecking output date {date}\033[0m (field valid_date={valid_date})")
            if (date, key) not in accumulators:
                print(f"  ❤️ Creating accumulator for date {date}, key {key}")
                accumulators[(date, key)] = Accumulator(date, period=period, key=key, coverage=coverages[date])

            # find the accumulator for this valid date and key
            acc = accumulators[(date, key)]

            # perform accumulation if needed
            if acc.compute(field, values, interval):
                # .compute() returned True, meaning the field was used for accumulation
                field_used = True
                print(f"    🆗️ Used for accumulator {acc.__repr__(verbose=True)  }")

                if acc.is_complete():
                    # all intervals for accumulation have been processed, write the accumulated field to output
                    acc.write_to_output(output, template=field)
                    print("   ✅ Completed : ", acc.__repr__(verbose=True))
                    print(f"  ✅ Wrote accumulated field for date {date}, key {key}")

        if not field_used:
            for a in accumulators.values():
                LOG.error(f"Existing accumulator: {a.__repr__(verbose=True)}")
            raise ValueError(f"Field {field} with {interval=} was provided by the source but not used")

    # Final checks
    def check_missing_accumulators():
        for date in dates:
            count = sum(1 for (d, k) in accumulators.keys() if d == date)
            LOG.debug(f"Date {date} has {count} accumulators")
            if count != len(accumulators) // len(dates):
                LOG.error(f"All requested dates: {dates}")
                LOG.error(f"Date {date} has {count} accumulators, expected {len(accumulators) // len(dates)}")
                for d, k in accumulators.keys():
                    if d == date:
                        LOG.error(f"  Accumulator for date {d}, key {k}")
                raise ValueError(f"Date {date} has {count} accumulators, expected {len(accumulators) // len(dates)}")

    check_missing_accumulators()

    for acc in accumulators.values():
        if not acc.is_complete():
            raise ValueError(f"Accumulator not complete: {acc.__repr__(verbose=True)}")

    print(f"Created {len(accumulators)} accumulated fields:")

    if not accumulators:
        raise ValueError("No accumulators were created, cannot produce accumulated datasource")

    output.close()
    ds = earthkit.data.from_source("file", path)
    ds._keep_file = tmp  # prevent deletion of temp file until ds is deleted

    print(f"Created {len(ds)} accumulated fields:")
    for i in ds:
        print("  ", i)
    return ds


@source_registry.register("accumulate")
class AccumulateSource(LegacySource):

    @staticmethod
    def _execute(
        context: Any,
        dates: list[datetime.datetime],
        source: Any,
        period,
        available=None,
        patch: Any = None,
        **kwargs,
    ) -> Any:
        """Accumulation source callable function.
        Read the recipe for accumulation in the request dictionary, check main arguments and call computation.

        Parameters:
        ----------
        context: Any,
            The dataset building context (will be updated with trace of accumulation)
        dates: list[datetime.datetime]
            The list of valid dates on which to perform accumulations.
        source: Any,
            The accumulation source
        period: str | int | datetime.timedelta,
            The interval over which to accumulate (user-defined)
        available: Any, optional
            A description of the available periods in the data source. See documentation.
        skip_checks: Any, optional
            Lots of metadata is checked during accumulations. This will prevent computing accumulation when
            the source is providing data with missing of wrong metadata. Some checks can be skipped
            to allow dataset creation despite inconsistent metadata.

        Return
        ------
        The accumulated data source.

        """
        if "skip_checks" in kwargs:
            raise ValueError("skip_checks is not supported anymore, use patch instead (not implemented).")

        if patch is not None:
            # patch will patch the fields returned by the source to fix metadata issues
            # such as missing base time, wrong stepType, or startStep/endStep swapped, or startStep=endStep.
            # this is not implemented yet but is required to handle some well-known cases.
            # The user should provide a patch description as a dictionary (API to be defined).
            raise NotImplementedError("patch is not implemented yet for accumulate source.")
            # patch will patch the fields returned by the source to fix metadata issues
            # such as missing base time, wrong stepType, or startStep/endStep swapped, or startStep=endStep.
            # this is not implemented yet but is required to handle some well-known cases.
            # The user should provide a patch to apply

        if "accumulation_period" in source:
            raise ValueError("'accumulation_period' should be define outside source for accumulate action as 'period'")
        period = frequency_to_timedelta(period)
        return _compute_accumulations(context, dates, source=source, period=period, available=available, **kwargs)
