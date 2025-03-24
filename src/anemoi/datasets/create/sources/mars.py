# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import re
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Union

from anemoi.utils.humanize import did_you_mean
from earthkit.data import from_source
from earthkit.data.utils.availability import Availability

from anemoi.datasets.create.utils import to_datetime_list

from .legacy import legacy_source

DEBUG = False


def to_list(x: Union[list, tuple, Any]) -> list:
    """Converts the input to a list if it is not already a list or tuple.

    Parameters
    ----------
    x : Any
        The input value to be converted.

    Returns
    -------
    list
        A list containing the input value(s).
    """
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def _date_to_datetime(
    d: Union[datetime.datetime, list, tuple, str],
) -> Union[datetime.datetime, List[datetime.datetime]]:
    """Converts the input date(s) to datetime objects.

    Parameters
    ----------
    d : Union[datetime.datetime, list, tuple, str]
        The input date(s) to be converted.

    Returns
    -------
    Union[datetime.datetime, List[datetime.datetime]]
        A datetime object or a list of datetime objects.
    """
    if isinstance(d, datetime.datetime):
        return d
    if isinstance(d, (list, tuple)):
        return [_date_to_datetime(x) for x in d]
    return datetime.datetime.fromisoformat(d)


def expand_to_by(x: Union[str, int, list]) -> Union[str, int, list]:
    """Expands a range expression to a list of values.

    Parameters
    ----------
    x : Union[str, int, list]
        The input range expression.

    Returns
    -------
    Union[str, int, list]
        A list of expanded values.
    """
    if isinstance(x, (str, int)):
        return expand_to_by(str(x).split("/"))

    if len(x) == 3 and x[1] == "to":
        start = int(x[0])
        end = int(x[2])
        return list(range(start, end + 1))

    if len(x) == 5 and x[1] == "to" and x[3] == "by":
        start = int(x[0])
        end = int(x[2])
        by = int(x[4])
        return list(range(start, end + 1, by))

    return x


def normalise_time_delta(t: Union[datetime.timedelta, str]) -> datetime.timedelta:
    """Normalizes a time delta string to a datetime.timedelta object.

    Parameters
    ----------
    t : Union[datetime.timedelta, str]
        The input time delta string.

    Returns
    -------
    datetime.timedelta
        A normalized datetime.timedelta object.
    """
    if isinstance(t, datetime.timedelta):
        assert t == datetime.timedelta(hours=t.hours), t

    assert t.endswith("h"), t

    t = int(t[:-1])
    t = datetime.timedelta(hours=t)
    return t


def _normalise_time(t: Union[int, str]) -> str:
    """Normalizes a time value to a string in HHMM format.

    Parameters
    ----------
    t : Union[int, str]
        The input time value.

    Returns
    -------
    str
        A string representing the normalized time.
    """
    t = int(t)
    if t < 100:
        t * 100
    return "{:04d}".format(t)


def _expand_mars_request(
    request: Dict[str, Any],
    date: datetime.datetime,
    request_already_using_valid_datetime: bool = False,
    date_key: str = "date",
) -> List[Dict[str, Any]]:
    """Expands a MARS request with the given date and other parameters.

    Parameters
    ----------
    request : Dict[str, Any]
        The input MARS request.
    date : datetime.datetime
        The date to be used in the request.
    request_already_using_valid_datetime : bool, optional
        Flag indicating if the request already uses valid datetime.
    date_key : str, optional
        The key for the date in the request.

    Returns
    -------
    List[Dict[str, Any]]
        A list of expanded MARS requests.
    """
    requests = []

    user_step = to_list(expand_to_by(request.get("step", [0])))
    user_time = None
    user_date = None

    if not request_already_using_valid_datetime:
        user_time = request.get("time")
        if user_time is not None:
            user_time = to_list(user_time)
            user_time = [_normalise_time(t) for t in user_time]

        user_date = request.get(date_key)
        if user_date is not None:
            assert isinstance(user_date, str), user_date
            user_date = re.compile("^{}$".format(user_date.replace("-", "").replace("?", ".")))

    for step in user_step:
        r = request.copy()

        if not request_already_using_valid_datetime:

            if isinstance(step, str) and "-" in step:
                assert step.count("-") == 1, step

            # this takes care of the cases where the step is a period such as 0-24 or 12-24
            hours = int(str(step).split("-")[-1])

            base = date - datetime.timedelta(hours=hours)
            r.update(
                {
                    date_key: base.strftime("%Y%m%d"),
                    "time": base.strftime("%H%M"),
                    "step": step,
                }
            )

        for pproc in ("grid", "rotation", "frame", "area", "bitmap", "resol"):
            if pproc in r:
                if isinstance(r[pproc], (list, tuple)):
                    r[pproc] = "/".join(str(x) for x in r[pproc])

        if user_date is not None:
            if not user_date.match(r[date_key]):
                continue

        if user_time is not None:
            # It time is provided by the user, we only keep the requests that match the time
            if r["time"] not in user_time:
                continue

        requests.append(r)

    # assert requests, requests

    return requests


def factorise_requests(
    dates: List[datetime.datetime],
    *requests: Dict[str, Any],
    request_already_using_valid_datetime: bool = False,
    date_key: str = "date",
) -> Generator[Dict[str, Any], None, None]:
    """Factorizes the requests based on the given dates.

    Parameters
    ----------
    dates : List[datetime.datetime]
        The list of dates to be used in the requests.
    requests : Dict[str, Any]
        The input requests to be factorized.
    request_already_using_valid_datetime : bool, optional
        Flag indicating if the requests already use valid datetime.
    date_key : str, optional
        The key for the date in the requests.

    Returns
    -------
    Generator[Dict[str, Any], None, None]
        Factorized requests.
    """
    updates = []
    for req in requests:
        # req = normalise_request(req)

        for d in dates:
            updates += _expand_mars_request(
                req,
                date=d,
                request_already_using_valid_datetime=request_already_using_valid_datetime,
                date_key=date_key,
            )

    if not updates:
        return

    compressed = Availability(updates)
    for r in compressed.iterate():
        for k, v in r.items():
            if isinstance(v, (list, tuple)) and len(v) == 1:
                r[k] = v[0]
        yield r


def use_grib_paramid(r: Dict[str, Any]) -> Dict[str, Any]:
    """Converts the parameter short names to GRIB parameter IDs.

    Parameters
    ----------
    r : Dict[str, Any]
        The input request containing parameter short names.

    Returns
    -------
    Dict[str, Any]
        The request with parameter IDs.
    """
    from anemoi.utils.grib import shortname_to_paramid

    params = r["param"]
    if isinstance(params, str):
        params = params.split("/")
    assert isinstance(params, (list, tuple)), params

    params = [shortname_to_paramid(p) for p in params]
    r["param"] = "/".join(str(p) for p in params)

    return r


MARS_KEYS = [
    "accuracy",
    "activity",
    "anoffset",
    "area",
    "bitmap",
    "channel",
    "class",
    "database",
    "dataset",
    "date",
    "diagnostic",
    "direction",
    "domain",
    "expect",
    "experiment",
    "expver",
    "fcmonth",
    "fcperiod",
    "fieldset",
    "filter",
    "format",
    "frame",
    "frequency",
    "gaussian",
    "generation",
    "grid",
    "hdate",
    "ident",
    "instrument",
    "interpolation",
    "intgrid",
    "iteration",
    "level",
    "levelist",
    "levtype",
    "method",
    "model",
    "month",
    "number",
    "obsgroup",
    "obstype",
    "offsetdate",
    "offsettime",
    "optimise",
    "origin",
    "packing",
    "padding",
    "param",
    "quantile",
    "realization",
    "reference",
    "reportype",
    "repres",
    "resol",
    "resolution",
    "rotation",
    "step",
    "stream",
    "system",
    "target",
    "time",
    "truncation",
    "type",
    "year",
]


@legacy_source(__file__)
def mars(
    context: Any,
    dates: List[datetime.datetime],
    *requests: Dict[str, Any],
    request_already_using_valid_datetime: bool = False,
    date_key: str = "date",
    use_cdsapi_dataset: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Executes MARS requests based on the given context, dates, and other parameters.

    Parameters
    ----------
    context : Any
        The context for the requests.
    dates : List[datetime.datetime]
        The list of dates to be used in the requests.
    requests : Dict[str, Any]
        The input requests to be executed.
    request_already_using_valid_datetime : bool, optional
        Flag indicating if the requests already use valid datetime.
    date_key : str, optional
        The key for the date in the requests.
    use_cdsapi_dataset : Optional[str], optional
        The dataset to be used with CDS API.
    kwargs : Any
        Additional keyword arguments for the requests.

    Returns
    -------
    Any
        The resulting dataset.
    """

    if not requests:
        requests = [kwargs]

    for r in requests:
        param = r.get("param", [])
        if not isinstance(param, (list, tuple)):
            param = [param]
        # check for "Norway bug" where yaml transforms 'no' into False, etc.
        for p in param:
            if p is False:
                raise ValueError(
                    "'param' cannot be 'False'. If you wrote 'param: no' or 'param: off' in yaml, you may want to use quotes?"
                )
            if p is None:
                raise ValueError(
                    "'param' cannot be 'None'. If you wrote 'param: no' in yaml, you may want to use quotes?"
                )
            if p is True:
                raise ValueError(
                    "'param' cannot be 'True'. If you wrote 'param: on' in yaml, you may want to use quotes?"
                )

    if len(dates) == 0:  # When using `repeated_dates`
        assert len(requests) == 1, requests
        assert "date" in requests[0], requests[0]
        if isinstance(requests[0]["date"], datetime.date):
            requests[0]["date"] = requests[0]["date"].strftime("%Y%m%d")
    else:
        requests = factorise_requests(
            dates,
            *requests,
            request_already_using_valid_datetime=request_already_using_valid_datetime,
            date_key=date_key,
        )

    requests = list(requests)

    ds = from_source("empty")
    context.trace("✅", f"{[str(d) for d in dates]}")
    context.trace("✅", f"Will run {len(requests)} requests")
    for r in requests:
        r = {k: v for k, v in r.items() if v != ("-",)}
        context.trace("✅", f"mars {r}")

    for r in requests:
        r = {k: v for k, v in r.items() if v != ("-",)}

        if context.use_grib_paramid and "param" in r:
            r = use_grib_paramid(r)

        for k, v in r.items():
            if k not in MARS_KEYS:
                raise ValueError(
                    f"⚠️ Unknown key {k}={v} in MARS request. Did you mean '{did_you_mean(k, MARS_KEYS)}' ?"
                )
        try:
            if use_cdsapi_dataset:
                ds = ds + from_source("cds", use_cdsapi_dataset, r)
            else:
                ds = ds + from_source("mars", **r)
        except Exception as e:
            if "File is empty:" not in str(e):
                raise
    return ds


execute = mars


if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(
        """
    - class: ea
      expver: '0001'
      grid: 20.0/20.0
      levtype: sfc
      param: [2t]
      # param: [10u, 10v, 2d, 2t, lsm, msl, sdor, skt, slor, sp, tcw, z]
      number: [0, 1]

    # - class: ea
    #   expver: '0001'
    #   grid: 20.0/20.0
    #   levtype: pl
    #   param: [q]
    #   levelist: [1000, 850]

    """
    )
    dates = yaml.safe_load("[2022-12-30 18:00, 2022-12-31 00:00, 2022-12-31 06:00, 2022-12-31 12:00]")
    dates = to_datetime_list(dates)

    DEBUG = True
    for f in mars(None, dates, *config):
        print(f, f.to_numpy().mean())
