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
from copy import deepcopy
from typing import Any

import numpy as np
from .accumulation_utils import utils
from .accumulation_utils import timelines as tl
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.create.utils import to_datetime_list

from .legacy import legacy_source

LOG = logging.getLogger(__name__)

xprint = print

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

    if timeline_class != tl.DefaultTimeline:
        _ = request.pop("data_accumulation_period")
        LOG.warning("Non-default data Timeline (e.g MARS): ignoring data_accumulation_period")

    return request, param, number

class Accumulator:
    values = None

    def __init__(
        self, 
        timeline_class: type[tl.Timeline],
        valid_date: datetime.datetime, 
        user_accumulation_period: datetime.timedelta,
        data_accumulation_period: datetime.timedelta,
        **kwargs):
        
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

        self.timeline = timeline_class(self.valid_date, user_accumulation_period, data_accumulation_period, **kwargs)
        
    @property
    def requests(self) -> dict:
        for period in self.timeline:
            # build the full data requests, merging the time requests with the key
            yield {**self.kwargs.copy(), **dict(period.time_request)}

    def is_field_needed(self, field):
        for k, v in self.key.items():
            metadata = field.metadata(k) if k != "number" else _member(field)
            if metadata != v:
                LOG.debug(f"{self} does not need field {field} because of {k}={metadata} not {v}")
                return False
        return True

    def compute(self, field: Any, values: NDArray) -> None:
        """
        Verify the field time metadata, find the associated period
        and perform accumulation with the values array on this 

        In any case values have been extracted from field

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

        xprint(f"{self} field ✅ ({period.sign}) {field} for {period}")

        # actual accumulation computation
        self.values = period.apply(self.values, values)
        self.timeline.set_done(period)

        if self.timeline.all_done():
            # all periods for accumulation have been processed
            # final list of outputs is ready to be updated
            self.write(field)  # field is used as a template
            xprint("accumulator", self, " : data written ✅ ")

    def write(self, field: Any) -> None:
        """
        Writing output inside a new field at the end of the accumulation
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
    action: Any,
    request: dict[str, Any],
    user_accumulation_period: datetime.timedelta,
    data_accumulation_period: datetime.timedelta,
    ) -> Any:

    timeline_class = tl.find_timeline_class(request)

    request, param, number = _prep_request(request, timeline_class)
    
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
                        **request,
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
    action.source.kwargs = {"request_already_using_valid_datetime": True, "shift_time_request": True}
    action.source.args = requests
    action.source.context = context
    # get the data (requests are packed to make a minimal number of queries to database)
    ds_to_accum = action.source.execute(dates)

    assert len(ds_to_accum) / len(param) / len(number) == len(overlapping_timelines), (
        f"retrieval yields {len(ds_to_accum)} fields, {len(param)} params, {len(number)} members ",
        f"but total number of periods requested is {len(overlapping_timelines)}",
        f"❌❌❌ error in {action.source}",
    )

    # send each field to each accumulator
    # the field will be used only if the accumulator has requested it
    for field in ds_to_accum:
        values = field.values  # optimisation : reading values only once
        for a in accumulators:
            a.compute(field, values)

    for a in accumulators:
        assert (a.timeline.all_done()), f"missing periods for accumulator {a}"
    
    # final data source
    ds = new_fieldlist_from_list([a.out for a in accumulators])
    
    # the resulting datasource has one field per valid date, parameter and ensemble member
    assert len(ds) / len(param) / len(number) == len(dates), (
        len(ds),
        len(param),
        len(dates),
    )

    return ds

@legacy_source(__file__)
def accumulations(context, dates, action, **request):
    utils._to_list(request["param"])
    try:
        user_accumulation_period = request.pop("accumulation_period")
        data_accumulation_period = request.pop("data_accumulation_period", "1h")
    except KeyError:
        raise ValueError("Accumulate action should provide 'accumulation_period', but none was found.")

    context.trace("🌧️", f"accumulations {request} {user_accumulation_period}")

    return _compute_accumulations(
        context,
        dates,
        action,
        request,
        user_accumulation_period=frequency_to_timedelta(user_accumulation_period),
        data_accumulation_period=frequency_to_timedelta(data_accumulation_period)
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
    dates = to_datetime_list(dates)

    class Context:
        use_grib_paramid = True

        def trace(self, *args):
            print(*args)

    for f in accumulations(Context, dates, **config):
        print(f, f.to_numpy().mean())
