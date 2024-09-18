# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import logging

from earthkit.data.core.fieldlist import MultiFieldList

from anemoi.datasets.create.functions.sources.mars import mars

LOGGER = logging.getLogger(__name__)
DEBUG = True


def _to_list(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


class HindcastCompute:
    def __init__(self, base_times, available_steps, request, reference_year):
        if not isinstance(base_times, (list, tuple)):
            base_times = [base_times]
        self.base_times = base_times
        self.available_steps = available_steps
        self.request = request

        # The user set 'hindcasts: True' in the 'dates' block of the configuration file.
        self.reference_year = reference_year
        if reference_year is None:
            # assert self.available_steps == [0], self.available_steps
            assert self.base_times == [0], self.base_times

    def compute_hindcast(self, date):
        if self.reference_year is None:
            return self._compute_hindcast1(date)
        else:
            return self._compute_hindcast2(date)

    def _compute_hindcast1(self, date):

        step = date.step
        hdate = datetime.datetime(date.year, date.refdate.month, date.refdate.day)
        r = self.request.copy()
        r["date"] = date.refdate
        r["hdate"] = hdate.strftime("%Y-%m-%d")
        r["time"] = 0
        r["step"] = step
        return r

    def _compute_hindcast2(self, date):
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
    if reference_year is None:
        return request, True

    request = request.copy()
    hdate = request.pop("date")

    if hdate.year >= reference_year:
        return None, False

    try:
        date = datetime.datetime(reference_year, hdate.month, hdate.day)
    except ValueError:
        if hdate.month == 2 and hdate.day == 29:
            return None, False
        raise

    request.update(date=date.strftime("%Y-%m-%d"), hdate=hdate.strftime("%Y-%m-%d"))
    return request, True


def hindcasts(context, dates, **request):

    from anemoi.datasets.dates import HindcastsDates

    provider = context.date_provider
    assert isinstance(provider, HindcastsDates)

    context.trace("H️", f"hindcasts {len(dates)=}")

    request["param"] = _to_list(request["param"])
    request["step"] = _to_list(request.get("step", 0))
    request["step"] = [int(_) for _ in request["step"]]

    context.trace("H️", f"hindcast {request}")

    requests = []
    for d in dates:
        r = request.copy()
        refdate, step = provider.mapping[d]
        r["hdate"] = d.strftime("%Y-%m-%d")
        r["date"] = refdate.strftime("%Y-%m-%d")
        r["time"] = refdate.strftime("%H")
        r["step"] = step
        requests.append(r)

    if len(requests) == 0:
        # print("HINDCASTS no requests")
        return MultiFieldList([])

    print("HINDCASTS requests", requests[0])

    print("HINDCASTS requests", len(requests))

    return mars(
        context,
        dates,
        *requests,
        date_key="hdate",
        request_already_using_valid_datetime=True,
    )


execute = hindcasts
