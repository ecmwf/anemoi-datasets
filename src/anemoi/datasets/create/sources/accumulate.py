# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import Any

import numpy as np
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.create.sources import source_registry

from .accumulate_utils.catalogues import Link
from .accumulate_utils.catalogues import build_catalogue
from .accumulate_utils.covering_intervals import SignedInterval
from .legacy import LegacySource

LOG = logging.getLogger(__name__)


class Accumulator:
    values: NDArray | None = None
    completed: bool = False

    def __init__(self, interval: SignedInterval, key: dict[str, Any], coverage):
        """Accumulator object for a given param/member/valid_date"""
        self.interval = interval
        self.coverage = coverage
        self.todo = [v for v in coverage]
        self.done = []
        assert interval.end > interval.start, f"Invalid interval {interval}"

        self.period = interval.max - interval.min
        self.valid_date = interval.max

        assert isinstance(key, dict)
        self.key = key
        for k in ["date", "time", "step"]:
            if k in self.key:
                raise ValueError(f"Cannot use {k} for accumulation")

        self.values = None  # will hold accumulated values array

        # The accumulator only accumulates fields matching its metadata
        # and does not know about the rest

        # key contains the mars request parameters except the one related to the time
        # A mars request is a dictionary with three categories of keys:
        #   - the ones related to the time (date, time, step)
        #   - the ones related to the data (param, stream, levtype, expver, number, ...)
        #   - the ones related to the processing to be done (grid, area, ...)

    def is_field_needed(self, field: Any):
        """Check whether the given field is needed by the accumulator (correct param, etc...)"""
        for k, v in self.key.items():

            if k == "number":
                continue

            metadata = field.metadata(k)
            if k == "number":  # bug in eccodes has number=None randomly
                if metadata is None:
                    metadata = 0
                if v is None:
                    v = 0
            if metadata != v:
                LOG.debug(f"{self} does not need field {field} because of {k}={metadata} not {v}")
                return False
        return True

    def is_complete(self, **kwargs) -> bool:
        """Check whether the accumulation is complete (all intervals have been processed)"""
        return not self.todo

    def compute(self, field: Any, values: NDArray, link: Link) -> None:
        """Verify the field time metadata, find the associated interval
        and perform accumulation with the values array on this interval.
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
        interval = link.interval
        if self.completed:
            raise ValueError(f"Accumulator {self} already completed, cannot process interval {interval}")

        # actual accumulation computation
        if interval.end < interval.start:  # negative accumulation if interval is reversed
            values = -values
        if self.values is None:
            self.values = values.copy()
        else:
            self.values += values.copy()

        if interval not in self.todo:
            raise ValueError(f"SignedInterval {interval} not in todo list of accumulator {self}")
        self.todo.remove(interval)
        self.done.append(interval)

    def field_to_interval(self, field: Any):
        valid_date = field.metadata("valid_date")
        step = field.metadata("step")
        date = valid_date - datetime.timedelta(hours=step)
        return SignedInterval(date=date, step=step)

    def get_accumulated_field(self, field: Any) -> None:
        """Writing output inside a new field at the end of the accumulation.
        The field is simply used as a template.

        Parameters:
        ----------
        field: Any
            An earthkit-data-like field

        Return
        ------
        None
        """
        assert self.is_complete(), self.todo

        # negative values may be an anomaly (e.g precipitation), but this is user's choice
        if np.any(self.values < 0):
            LOG.warning(
                f"Negative values when computing accumutation for {self}): min={np.amin(self.values)} max={np.amax(self.values)}"
            )

        self.completed = True
        return new_accumulated_field_with_valid_time(
            template=field,
            values=self.values,
            valid_date=self.valid_date,
            period=self.interval.end - self.interval.start,
        )

    def __repr__(self):
        key = ", ".join(f"{k}={v}" for k, v in self.key.items())
        return f"{self.__class__.__name__}({self.interval}, {key}, {len(self.coverage)-len(self.todo)}/{len(self.coverage)} already accumulated)"


def new_accumulated_field_with_valid_time(
    template, values, valid_date: datetime.datetime, period: datetime.timedelta
) -> Any:
    MISSING_VALUE = 1e-38
    assert np.all(values != MISSING_VALUE)

    startStep = datetime.timedelta(hours=0)
    endStep = period

    hours = endStep.total_seconds() / 3600

    if template.metadata("edition") == 1 and (hours > 254 or not hours.is_integer()):
        # this is a special case for GRIB edition 1 which only supports integer hours up to 254
        assert hours.is_integer(), f"edition 1 accumulation period must be integer hours, got {hours}"
        assert hours <= 254, f"edition 1 accumulation period must be <=254 hours, got {hours}"
        accumfield = new_field_from_numpy(
            values,
            template=template,
            stepType="instant",
            step=hours,
            check_nans=True,
            missing_value=MISSING_VALUE,
        )
    else:
        # this is the normal case for GRIB edition 2. And with edition 1 when hours are integer and <=254
        accumfield = new_field_from_numpy(
            values,
            template=template,
            stepType="accum",
            startStep=startStep,
            endStep=endStep,
            check_nans=True,
            missing_value=MISSING_VALUE,
        )
    return new_field_with_valid_datetime(accumfield, valid_date)


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

    cataloguer = build_catalogue(context, available, source, **kwargs)

    def interval_from_valid_date_and_period(valid_date, period):
        # helper function to build accumulation interval from valid date and period
        if not isinstance(valid_date, datetime.datetime):
            raise TypeError("valid_date must be a datetime.datetime instance")
        if not isinstance(period, datetime.timedelta):
            period = frequency_to_timedelta(period)
        start = valid_date - period
        end = valid_date
        return SignedInterval(start, end)

    # building accumulators :
    # one accumulator per valid date, per output parm and ensemble member
    # i.e. one accumulator per output field
    accumulators = []
    for valid_date in dates:
        interval = interval_from_valid_date_and_period(valid_date, period)
        coverage = cataloguer.covering_intervals(interval.start, interval.end)
        for key in cataloguer.get_all_keys():
            accumulators.append(Accumulator(interval, key=key, coverage=coverage))

    # building links from accumulators to intervals they need
    # each link represent a field to fetch from the source
    # i.e. one link per input field
    # one accumulator may generate multiple links if it needs multiple input fields
    # and one link may be used by multiple accumulators if they need the same input field
    links = []
    for accumulator in accumulators:
        LOG.debug(f"💬 {accumulator} will need:")
        for v in accumulator.coverage:
            assert isinstance(v, SignedInterval), type(v)
            link = Link(interval=v, accumulator=accumulator, catalogue=cataloguer)
            LOG.debug("  💬 ", link)
            links.append(link)

    out = []
    # for each field provided by the catalogue, find which accumulators need it and perform accumulation
    for field, values, link in cataloguer.retrieve_fields(links):
        link.accumulator.compute(field, values, link)
        if link.accumulator.is_complete():
            # all intervals for accumulation have been processed
            # final list of outputs is ready to be updated
            accumulated_field = link.accumulator.get_accumulated_field(field)  # field is used as a template
            out.append(accumulated_field)
    ds = new_fieldlist_from_list(out)

    # tmp = temp_file()
    # path = tmp.path
    # out = new_grib_output(path)
    # for f in ds:
    #     out.write(f)
    # out.close()
    # ds = ekd.from_source("file", path)

    # check that each accumulator is complete
    for link in links:
        a = link.accumulator
        if not a.is_complete():
            a.is_complete(debug=True)
            LOG.error(f"Accumulator incomplete: {a}")
            LOG.error(f"{len(a.done)} SignedInterval received:")
            for v in a.done:
                LOG.error(f"  {v}")
            raise AssertionError("missing periods for accumulator, stopping.")

    # the resulting datasource has one field per valid date, parameter and ensemble member
    keys = list(cataloguer.get_all_keys())
    assert len(ds) / len(keys) == len(dates), (len(ds), len(keys), len(dates))
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
        if "accumulation_period" in source:
            raise ValueError("'accumulation_period' should be define outside source for accumulate action as 'period'")
        period = frequency_to_timedelta(period)
        return _compute_accumulations(context, dates, source=source, period=period, available=available, **kwargs)
