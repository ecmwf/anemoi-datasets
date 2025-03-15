# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.create.sources.mars import mars

from .legacy import legacy_source

LOGGER = logging.getLogger(__name__)


def _to_list(x: Union[list, tuple, Any]) -> List[Any]:
    """Converts the input to a list if it is not already a list or tuple.

    Parameters
    ----------
    x : Any
        The input to convert.

    Returns
    -------
    list
        A list containing the input elements.
    """
    if isinstance(x, (list, tuple)):
        return x
    return [x]


@legacy_source(__file__)
def hindcasts(context: Any, dates: List[Any], **request: Dict[str, Any]) -> MultiFieldList:
    """Generates hindcast requests based on the provided dates and request parameters.

    Parameters
    ----------
    context : Any
        The context containing the dates provider and trace method.
    dates : List[Any]
        A list of dates for which to generate hindcast requests.
    request : Dict[str, Any]
        Additional request parameters.

    Returns
    -------
    MultiFieldList
        A MultiFieldList containing the hindcast data.
    """
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
