# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from types import SimpleNamespace

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.sources.grib_index import GribIndexSource


class DummyContext:
    def trace(self, *args, **kwargs):
        pass


class CapturingGribIndexSource(GribIndexSource):
    def __init__(self):
        super().__init__(
            DummyContext(),
            indexdb="/tmp/not-used.db",
            param=["2t"],
            levtype="sfc",
        )
        self.captured = None

    def _run_requests(self, full_requests):
        self.captured = full_requests
        return []


def test_grib_index_forecast_dates_builds_basetime_step_request_00utc():
    source = CapturingGribIndexSource()

    basetime = datetime.datetime(2021, 9, 1, 0)
    valid_time = datetime.datetime(2021, 9, 1, 6)

    source.execute_forecast_dates(ForecastDates([(valid_time, basetime)]))

    assert source.captured == [
        (
            [valid_time],
            {
                "param": ["2t"],
                "levtype": "sfc",
                "date": "20210901",
                "time": ["0", "0000"],
                "step": 6,
            },
        )
    ]


def test_grib_index_forecast_dates_builds_basetime_step_request_12utc():
    source = CapturingGribIndexSource()

    basetime = datetime.datetime(2021, 9, 1, 12)
    valid_time = datetime.datetime(2021, 9, 1, 18)

    source.execute_forecast_dates(ForecastDates([(valid_time, basetime)]))

    assert source.captured[0][1]["date"] == "20210901"
    assert source.captured[0][1]["time"] == "1200"
    assert source.captured[0][1]["step"] == 6


def test_grib_index_forecast_intervals_builds_basetime_step_request():
    source = CapturingGribIndexSource()

    basetime = datetime.datetime(2021, 9, 1, 0)
    valid_time = datetime.datetime(2021, 9, 1, 12)
    period = datetime.timedelta(hours=6)

    interval = SimpleNamespace(
        base=basetime,
        max=valid_time,
        valid_time=valid_time,
    )

    source.execute_forecast_intervals(
        ForecastIntervals(
            [(valid_time, basetime, period)],
            intervals=[interval],
        )
    )

    assert source.captured == [
        (
            [valid_time],
            {
                "param": ["2t"],
                "levtype": "sfc",
                "date": "20210901",
                "time": ["0", "0000"],
                "step": 12,
            },
        )
    ]
