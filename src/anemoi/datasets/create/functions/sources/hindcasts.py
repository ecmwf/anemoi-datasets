# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime

from anemoi.datasets.create.functions.sources.mars import mars

DEBUG = True


def _member(field):
    # Bug in eccodes has number=0 randomly
    number = field.metadata("number")
    if number is None:
        number = 0
    return number


def _to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


class HindcastCompute:
    def __init__(self, base_times, available_steps, request):
        self.base_times = base_times
        self.available_steps = available_steps
        self.request = request

    def compute_hindcast(self, date):
        result = []
        for step in sorted(self.available_steps):  # Use the shortest step
            start_date = date - datetime.timedelta(hours=step)
            hours = start_date.hour
            if hours in self.base_times:
                r = self.request.copy()
                r["date"] = start_date
                r["time"] = f"{start_date.hour:02d}00"
                r["step"] = step
                result.append(r)

        if not result:
            raise ValueError(
                f"Cannot find data for {self.request} for {date} (base_times={self.base_times}, "
                f"available_steps={self.available_steps})"
            )

        if len(result) > 1:
            raise ValueError(
                f"Multiple requests for {self.request} for {date} (base_times={self.base_times}, "
                f"available_steps={self.available_steps})"
            )

        return result[0]


def use_reference_year(reference_year, request):
    request = request.copy()
    hdate = request.pop("date")
    date = datetime.datetime(reference_year, hdate.month, hdate.day)
    request.update(date=date.strftime("%Y-%m-%d"), hdate=hdate.strftime("%Y-%m-%d"))
    return request


def hindcasts(context, dates, **request):
    request["param"] = _to_list(request["param"])
    request["step"] = _to_list(request["step"])
    request["step"] = [int(_) for _ in request["step"]]

    if request.get("stream") == "enfh" and "base_times" not in request:
        request["base_times"] = [0]

    available_steps = request.pop("step")
    available_steps = _to_list(available_steps)

    base_times = request.pop("base_times")

    reference_year = request.pop("reference_year")

    context.trace("H️", f"hindcast {request} {base_times} {available_steps} {reference_year}")

    c = HindcastCompute(base_times, available_steps, request)
    requests = []
    for d in dates:
        req = c.compute_hindcast(d)
        req = use_reference_year(reference_year, req)

        requests.append(req)

    return mars(
        context,
        dates,
        *requests,
        date_key="hdate",
        request_already_using_valid_datetime=True,
    )


execute = hindcasts
