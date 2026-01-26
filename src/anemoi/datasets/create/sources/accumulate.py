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
from .accumulate_utils.field_to_interval import FieldToInterval
from .legacy import LegacySource

LOG = logging.getLogger(__name__)


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
        step = int((interval.max - interval.base).total_seconds() / 3600)
        r["date"] = interval.base.strftime("%Y%m%d")
        r["time"] = interval.base.strftime("%H%M")
        r["step"] = step
        return interval.max, r, step


class IntervalsDatesProvider:
    def __init__(self, dates, coverages):
        self._dates = dates
        self.date_to_intervals = coverages

    def _adjust_request_to_interval(self, interval: Any, request: list[dict]) -> tuple[Any]:
        return _adjust_request_to_interval(interval, request)

    @property
    def intervals(self):
        for d in self._dates:
            for interval in self.date_to_intervals[d]:
                yield d, interval

    def __len__(self):
        return len(self._dates)

    def __iter__(self):
        yield from self._dates

    def __getitem__(self, index):
        return self._dates[index]


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

    def compute(self, values: NDArray, interval: SignedInterval) -> None:
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
                if i.min == interval.min and i.max == interval.max and i.base == interval.base:
                    return i
                if i.start == interval.start and i.end == interval.end and i.base is None:
                    return i
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
        local_values = matching.sign * values.copy()
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
        for k, v in self.key:
            if k == "param" and v == "tp":
                if np.any(self.values < 0):
                    LOG.warning(
                        f"Negative values when computing accumutation for {self}): min={np.nanmin(self.values)} max={np.nanmax(self.values)}"
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
    if hours != int(hours):
        raise ValueError(f"Accumulation period must be integer hours, got {hours}")
    hours = int(hours)

    if template.metadata("edition") == 1:
        # this is a special case for GRIB edition 1 which only supports integer hours up to 254
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


class Logs(list):
    def __init__(self, *args, accumulators, source, source_object, field_to_interval, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulators = accumulators
        self.source = source
        self.source_object = source_object
        self.field_to_interval = field_to_interval

    def raise_error(self, msg, field=None, field_interval=None) -> str:
        INTERVAL_COLOR = "\033[93m"
        FIELD_COLOR = "\033[92m"
        KEY_COLOR = "\033[95m"
        RESET_COLOR = "\033[0m"

        res = [""]
        res.append(f"❌ {msg}")
        res.append(f"💬 Patches applied: {self.field_to_interval.patches}")
        res.append("💬 Current field:")
        res.append(f" {FIELD_COLOR}{field}{RESET_COLOR}")
        res.append(f" {INTERVAL_COLOR}{field_interval}{RESET_COLOR}")
        if self.accumulators:
            res.append(f"💬 Existing accumulators ({len(self.accumulators)}) :")
            for a in self.accumulators.values():
                res.append(f"  {a.__repr__(verbose=True)}")
        res.append(f"💬 Received fields ({len(self)}):")
        for log in self:
            res.append(f"  {KEY_COLOR}{log[0]}{RESET_COLOR} {INTERVAL_COLOR}{log[2]}{RESET_COLOR}")
            res.append(f"       {KEY_COLOR}{log[1]}{RESET_COLOR}")
            for d, acc_repr in zip(log[3], log[4]):
                res.append(f"   used for date {d}: {acc_repr}")

        LOG.error("\n".join(res))
        res = ["More details below:"]

        res.append(f"💬 Fields returned to be accumulated ({len(self.source_object)}):")
        for field in self.source_object:
            res.append(
                f"  {field}, startStep={field.metadata('startStep')}, endStep={field.metadata('endStep')} mean={np.nanmean(field.values, axis=0)}"
            )

        LOG.error("\n".join(res))
        res = ["Even more details below:"]

        if "mars" in self.source:
            res.append("💬 Example of code fetching some available fields and inspect them:")
            res.append("# --------------------------------------------------")
            code = []
            code.append("from earthkit.data import from_source")
            code.append("import numpy as np")
            code.append('ds = from_source("mars", **{')
            for k, v in self.source["mars"].items():
                code.append(f"    {k!r}: {v!r},")
            code.append(f'    "date": {field.metadata("date")!r},')
            code.append(f'    "time": {field.metadata("time")!r}, # "ALL"')
            code.append(f'    "step": "ALL", # {field.metadata("step")!r},')
            code.append("})")
            code.append('print(f"Got {len(ds)} fields:")')
            code.append("prev_m = None")
            code.append("for field in ds[:50]: # limit to first 50 for brevity")
            code.append(
                '    print(f"{field} startStep={field.metadata("startStep")}, endStep={field.metadata("endStep")} mean={np.nanmean(field.values)}")'
            )
            res.append("# --------------------------------------------------")
            code.append("")
            res += code

            # now execute the code to show actual field values
            LOG.error("\n".join(res))

        raise ValueError(msg)


def _compute_accumulations(
    context: Any,
    dates: list[datetime.datetime],
    period: datetime.timedelta,
    source: dict,
    availability: dict[str, Any] | None = None,
    patch: dict | None = None,
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
    source: dict
        The source configuration to request fields from
    period: datetime.timedelta,
        The interval over which to accumulate (user-defined)
    availability: Any, optional
        A description of the available periods in the data source. See documentation.
    patch: list[dict] | None, optional
        A description of patches to apply to fields returned by the source to fix metadata issues.

    Return
    ------
    The accumulated datasource for all dates, parameters, members.

    """

    LOG.debug("💬 source for accumulations: %s", source)
    field_to_interval = FieldToInterval(patch)

    # building the source objects
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

    h = hashlib.md5(json.dumps((str(period), source), sort_keys=True).encode()).hexdigest()
    source_object = context.create_source(source, "data_sources", h)

    interval_generator = interval_generator_factory(availability, source_name, source[source_name])

    # generate the interval coverage for every date
    coverages = {}
    for d in dates:
        if not isinstance(d, datetime.datetime):
            raise TypeError("valid_date must be a datetime.datetime instance")
        coverages[d] = interval_generator.covering_intervals(d - period, d)
        LOG.debug(f"  Found covering intervals: for {d - period} to {d}:")
        for c in coverages[d]:
            LOG.debug(f"    {c}")

    intervals = IntervalsDatesProvider(dates, coverages)

    # need a temporary file to store the accumulated fields for now, because earthkit-data
    # does not completely support in-memory fieldlists yet (metadata consistency is not fully ensured)
    tmp = temp_file()
    path = tmp.path
    output = new_grib_output(path)

    accumulators = {}
    logs = Logs(
        accumulators=accumulators,
        source=source,
        source_object=source_object(context, intervals),
        field_to_interval=field_to_interval,
    )
    for field in source_object(context, intervals):
        # for each field provided by the catalogue, find which accumulators need it and perform accumulation

        values = field.values.copy()

        key = field.metadata(namespace="mars")
        key = {k: v for k, v in key.items() if k not in ["date", "time", "step","timespan"]}
        key = tuple(sorted(key.items()))
        log = " ".join(f"{k}={v}" for k, v in field.metadata(namespace="mars").items())

        field_interval = field_to_interval(field)

        logs.append([str(field), log, field_interval, [], []])

        if field_interval.end <= field_interval.start:
            logs.raise_error("Invalid field interval with end <= start", field=field, field_interval=field_interval)

        field_used = False
        for date in dates:
            # build accumulator if it does not exist yet
            if (date, key) not in accumulators:
                accumulators[(date, key)] = Accumulator(date, period=period, key=key, coverage=coverages[date])

            # find the accumulator for this valid date and key
            acc = accumulators[(date, key)]

            # perform accumulation if needed
            if acc.compute(values, field_interval):
                # .compute() returned True, meaning the field was used for accumulation
                field_used = True
                logs[-1][3].append(date)
                logs[-1][4].append(acc.__repr__(verbose=True))

                if acc.is_complete():
                    # all intervals for accumulation have been processed, write the accumulated field to output
                    acc.write_to_output(output, template=field)

        if not field_used:
            logs.raise_error("Field not used for any accumulation", field=field, field_interval=field_interval)

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

    LOG.info(f"Created {len(accumulators)} accumulated fields")

    if not accumulators:
        raise ValueError("No accumulators were created, cannot produce accumulated datasource")

    output.close()
    ds = earthkit.data.from_source("file", path)
    ds._keep_file = tmp  # prevent deletion of temp file until ds is deleted

    LOG.debug(f"Created {len(ds)} accumulated fields:")
    for i in ds:
        LOG.debug("  %s", i)
    return ds


@source_registry.register("accumulate")
class AccumulateSource(LegacySource):

    @staticmethod
    def _execute(
        context: Any,
        dates: list[datetime.datetime],
        source: Any,
        period: str | int | datetime.timedelta,
        availability=None,
        patch: Any = None,
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
        availability: Any, optional
            A description of the available periods in the data source. See documentation.
        patch: Any, optional
            A description of patches to apply to fields returned by the source to fix metadata issues.

        Return
        ------
        The accumulated data source.

        """
        if availability is None:
            raise ValueError(
                "Argument 'availability' must be specified for accumulate source. See https://anemoi.readthedocs.io/projects/datasets/en/latest/building/sources/accumulate.html"
            )

        if "accumulation_period" in source:
            raise ValueError("'accumulation_period' should be define outside source for accumulate action as 'period'")

        period = frequency_to_timedelta(period)
        return _compute_accumulations(
            context, dates, source=source, period=period, availability=availability, patch=patch
        )
