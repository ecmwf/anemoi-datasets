# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging

from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.create.functions.sources.mars import mars

LOGGER = logging.getLogger(__name__)


def _to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def hindcasts(context, dates, **request):

    from anemoi.datasets.dates import HindcastsDates

    provider = context.dates_provider
    assert isinstance(provider, HindcastsDates)

    context.trace("H️", f"hindcasts {len(dates)=}")

    request["param"] = _to_list(request["param"])
    request["step"] = _to_list(request.get("step", 0))
    request["step"] = [int(_) for _ in request["step"]]

    context.trace("H️", f"hindcast {request}")

    requests = []
    for d in dates:
        r = request.copy()
        hindcast = provider.mapping[d]
        r["hdate"] = hindcast.hdate.strftime("%Y-%m-%d")
        r["date"] = hindcast.refdate.strftime("%Y-%m-%d")
        r["time"] = hindcast.refdate.strftime("%H")
        r["step"] = hindcast.step
        requests.append(r)

    if len(requests) == 0:
        return MultiFieldList([])

    return mars(
        context,
        dates,
        *requests,
        date_key="hdate",
        request_already_using_valid_datetime=True,
    )


execute = hindcasts
