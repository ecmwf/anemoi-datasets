# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
from copy import deepcopy

from climetlab import load_source
from climetlab.utils.availability import Availability

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


def normalise_time_delta(t):
    if isinstance(t, datetime.timedelta):
        assert t == datetime.timedelta(hours=t.hours), t

    assert t.endswith("h"), t

    t = int(t[:-1])
    t = datetime.timedelta(hours=t)
    return t


def _expand_mars_request(request, date, date_key="date"):
    requests = []
    step = to_list(request.get("step", [0]))
    for s in step:
        r = deepcopy(request)

        if isinstance(s, str) and "-" in s:
            assert s.count("-") == 1, s
        # this takes care of the cases where the step is a period such as 0-24 or 12-24
        hours = int(str(s).split("-")[-1])

        base = date - datetime.timedelta(hours=hours)
        r.update(
            {
                date_key: base.strftime("%Y%m%d"),
                "time": base.strftime("%H%M"),
                "step": s,
            }
        )

        for pproc in ("grid", "rotation", "frame", "area", "bitmap", "resol"):
            if pproc in r:
                if isinstance(r[pproc], (list, tuple)):
                    r[pproc] = "/".join(str(x) for x in r[pproc])

        requests.append(r)

    return requests


def factorise_requests(dates, *requests, date_key="date"):
    updates = []
    for req in requests:
        # req = normalise_request(req)

        for d in dates:
            updates += _expand_mars_request(req, date=d, date_key=date_key)

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


def mars(context, dates, *requests, date_key="date", **kwargs):
    if not requests:
        requests = [kwargs]

    requests = factorise_requests(dates, *requests, date_key=date_key)
    ds = load_source("empty")
    for r in requests:
        r = {k: v for k, v in r.items() if v != ("-",)}

        if context.use_grib_paramid and "param" in r:
            r = use_grib_paramid(r)

        if DEBUG:
            context.trace("✅", f"load_source(mars, {r}")

        ds = ds + load_source("mars", **r)
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
