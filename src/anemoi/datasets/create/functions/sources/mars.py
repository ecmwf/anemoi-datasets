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

from anemoi.utils.humanize import did_you_mean
from earthkit.data import from_source
from earthkit.data.utils.availability import Availability

from anemoi.datasets.create.utils import to_datetime_list

DEBUG = False


def to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def _date_to_datetime(d):
    if isinstance(d, datetime.datetime):
        return d
    if isinstance(d, (list, tuple)):
        return [_date_to_datetime(x) for x in d]
    return datetime.datetime.fromisoformat(d)


def expand_to_by(x):

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


def normalise_time_delta(t):
    if isinstance(t, datetime.timedelta):
        assert t == datetime.timedelta(hours=t.hours), t

    assert t.endswith("h"), t

    t = int(t[:-1])
    t = datetime.timedelta(hours=t)
    return t


def _normalise_time(t):
    t = int(t)
    if t < 100:
        t * 100
    return "{:04d}".format(t)


def _expand_mars_request(request, date, request_already_using_valid_datetime=False, date_key="date"):
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
    dates,
    *requests,
    request_already_using_valid_datetime=False,
    date_key="date",
):
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


def use_grib_paramid(r):
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


def mars(
    context,
    dates,
    *requests,
    request_already_using_valid_datetime=False,
    date_key="date",
    use_cdsapi_dataset=None,
    **kwargs,
):

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
