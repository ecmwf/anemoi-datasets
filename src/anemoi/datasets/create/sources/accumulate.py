# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
from copy import deepcopy
from typing import Any

import md5
import numpy as np
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.create.sources import source_registry

from .accumulation_utils import timelines as tl
from .accumulation_utils import utils
from .legacy import LegacySource

LOG = logging.getLogger(__name__)


def _prep_request(request: dict[str, Any], timeline_class: type[tl.Timeline]) -> dict[str, Any]:
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


class Accumulator:
    values: NDArray | None = None

    def __init__(
        self,
        timeline_class: type[tl.Timeline],
        valid_date: datetime.datetime,
        user_accumulation_period: datetime.timedelta,
        data_accumulation_period: datetime.timedelta,
        **kwargs: dict,
    ):
        """Accumulator object for a given param/member/valid_date

        Parameters:
        ---------
        timeline_class: type[tl.Timeline]
            The type of Timeline that should be used by this accumulator (depends on data source)
        valid_date: datetime.datetime
            the valid date at which the Accumulator refers (final value will indicate accumulation up to this date)
        user_accumulation_period: datetime.timedelta
            User-defined accumulation period
        data_accumulation_period: datetime.timedelta,
            Source data accumulation period
        **kwargs: dict
            Additional kwargs coming from accumulation recipe
        """

        self.valid_date = valid_date

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

        # instantiate Timeline object
        if timeline_class != tl.DefaultTimeline:
            LOG.warning("Non-default data Timeline (e.g MARS): ignoring data_accumulation_period")
            data_accumulation_period = frequency_to_timedelta("1h")  # only to ensure compatibility
        self.timeline = timeline_class(self.valid_date, user_accumulation_period, data_accumulation_period, **kwargs)

    @property
    def requests(self) -> dict:
        """build the full data requests, merging the time requests with the key.
        This will be used to query the source data database.
        """
        for period in self.timeline:
            yield {**self.kwargs.copy(), **dict(period.time_request)}

    def is_field_needed(self, field: Any):
        """Check whether the given field is needed by the accumulator (correct param, etc...)"""
        for k, v in self.key.items():
            metadata = field.metadata(k) if k != "number" else utils._member(field)
            if metadata != v:
                LOG.debug(f"{self} does not need field {field} because of {k}={metadata} not {v}")
                return False
        return True

    def compute(self, field: Any, values: NDArray) -> None:
        """Verify the field time metadata, find the associated period
        and perform accumulation with the values array on this period.
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

        # check if field has correct parameters for the Accumulator (param, number)
        if not self.is_field_needed(field):
            return

        period = self.timeline.find_matching_period(field)
        if not period:
            return

        # each field must be seen once
        assert self.timeline.is_todo(period), (self.timeline, period)
        assert not self.timeline.is_done(period), f"Field {field} for period {period} already done"

        print(f"{self} field ✅ ({period.sign}) {field} for {period}")

        # actual accumulation computation
        self.values = period.apply(self.values, values)
        self.timeline.set_done(period)

        if self.timeline.all_done():
            # all periods for accumulation have been processed
            # final list of outputs is ready to be updated
            self.write(field)  # field is used as a template
            print("accumulator", self, " : data written ✅ ")

    def write(self, field: Any) -> None:
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
        assert self.timeline.all_done(), self.timeline

        # negative values may be an anomaly (e.g precipitation), but this is user's choice
        if np.any(self.values < 0):
            LOG.warning(
                f"Negative values when computing accumutation for {self}): min={np.amin(self.values)} max={np.amax(self.values)}"
            )

        startStep = datetime.timedelta(hours=0)
        endStep = self.timeline.accumulation_period

        accumfield = new_field_from_numpy(
            self.values, template=field, startStep=startStep, endStep=endStep, stepType="accum"
        )

        self.out = new_field_with_valid_datetime(accumfield, self.valid_date)

        # resetting values as accumulation is done
        self.values = None

    def __repr__(self):
        key = ", ".join(f"{k}={v}" for k, v in self.key.items())
        return f"{self.__class__.__name__}({self.valid_date}, {key})"


def _compute_accumulations(
    context: Any,
    dates: list[datetime.datetime],
    source_name: Any,
    source_request: dict[str, Any],
    user_accumulation_period: datetime.timedelta,
    data_accumulation_period: datetime.timedelta,
) -> Any:
    """Concrete accumulation logic.

    - identify the needed timelines for each date/parameter/member defined in recipe
    - fetch the source data via a database (mars or grib-index)
    - create Accumulator objects and fill them will accumulated values from source data
    - return the datasource with accumulated values

    Parameters:
    ----------
    context: Any,
        The dataset building context (will be updated with trace of accumulation)
    dates: list[datetime.datetime]
        The list of valid dates on which to perform accumulations.
    source_name: Any,
        The abstract AccumulationAction object containing the accumulation source
    source_request: dict[str,Any]
        The parameters from the accumulation recipe, except for user/data accumulation periods
    user_accumulation_period: datetime.timedelta,
        The interval over which to accumulate (user-defined)
    data_accumulation_period: datetime.timedelta
        The interval over which source data is already accumulated (default 1h if not user-specified, ignored for mars data).

    Return
    ------
    The accumulated datasource for all dates, parameters, members.

    """

    # timeline class depends on data source ; split between Era-like timelines and Default ones
    timeline_class = tl.find_timeline_class(source_request)

    source_request, param, number = _prep_request(source_request, timeline_class)

    # building accumulators
    accumulators = []
    # some valid dates might have identical/overlapping timelines
    overlapping_timelines = set()

    # one accumulator per valid date, output field and ensemble member
    for valid_date in dates:
        for p in param:
            for n in number:
                accumulators.append(
                    Accumulator(
                        timeline_class,
                        valid_date,
                        user_accumulation_period=user_accumulation_period,
                        data_accumulation_period=data_accumulation_period,
                        param=p,
                        number=n,
                        **source_request,
                    )
                )

        # this is the exact number of periods that should be retrieved from action.source
        overlapping_timelines.update({period for period in accumulators[-1].timeline})

    # get all needed data requests
    requests = []
    for a in accumulators:
        for r in a.requests:
            requests.append(r)

    # these arguments are needed for the database to retrieve the fields with the right valid time

    source = context.create_source(
        dict(source_name=dict(requests=requests, request_already_using_valid_datetime=True, shift_time_request=True)),
        "data_sources",
        "accumulate",
        md5(json.dumps([source_name, requests], sort_keys=True).encode()).hexdigest(),
    )

    # get the data (requests are packed to make a minimal number of queries to database)
    ds_to_accum = source.execute(dates)

    assert len(ds_to_accum) / len(param) / len(number) == len(overlapping_timelines), (
        f"retrieval yields {len(ds_to_accum)} fields, {len(param)} params, {len(number)} members ",
        f"but total number of periods requested is {len(overlapping_timelines)}",
        f"❌❌❌ error in {source_name}",
    )

    # send each field to each accumulator
    # the field will be used only if the accumulator has requested it
    for field in ds_to_accum:
        values = field.values  # optimisation : reading values only once
        for a in accumulators:
            a.compute(field, values)

    for a in accumulators:
        assert a.timeline.all_done(), f"missing periods for accumulator {a}"

    # final data source
    ds = new_fieldlist_from_list([a.out for a in accumulators])

    # the resulting datasource has one field per valid date, parameter and ensemble member
    assert len(ds) / len(param) / len(number) == len(dates), (
        len(ds),
        len(param),
        len(dates),
    )

    return ds


@source_registry.register("accumulate")
class Accumulations2Source(LegacySource):

    @staticmethod
    def _execute(
        context: Any, dates: list[datetime.datetime], source: Any, accumulation_period, data_accumulation_period="1h"
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
        request: dict[str,Any]
            The parameters from the accumulation recipe

        Return
        ------
        The accumulated data source.

        """

        assert isinstance(source, dict)
        assert len(source) == 1
        assert "param" not in source, "param should be defined inside source for accumulate action"

        user_accumulation_period = accumulation_period

        source_name, source_request = next(iter(source.items()))
        source_request = source_request.copy()

        return _compute_accumulations(
            context,
            dates,
            source_name,
            source_request,
            user_accumulation_period=frequency_to_timedelta(user_accumulation_period),
            data_accumulation_period=frequency_to_timedelta(data_accumulation_period),
        )
