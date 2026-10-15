# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A source that cannot address a field by its interval says so, instead of falling back."""

import datetime

import pytest

from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.source import Source

BASE = datetime.datetime(2021, 1, 1)
END = BASE + datetime.timedelta(hours=6)
INTERVAL = SignedInterval(start=BASE, end=END, base=BASE)


class _InstantOnly(Source):
    """Serves validity dates and runs, as most sources do -- but no intervals."""

    def execute_valid_dates(self, argument):
        raise AssertionError("must not fall back to execute_valid_dates")

    def execute_forecast_dates(self, argument):
        raise AssertionError("must not fall back to execute_forecast_dates")


def test_intervals_are_refused_rather_than_read_as_validity_dates() -> None:
    with pytest.raises(NotImplementedError, match="cannot serve archived intervals: .*'fdb'"):
        _InstantOnly(context=None).execute(Intervals([END], [INTERVAL]))


def test_forecast_intervals_are_refused_rather_than_read_as_runs() -> None:
    argument = ForecastIntervals(items=[(END, BASE, END - BASE)], intervals=[INTERVAL])
    with pytest.raises(NotImplementedError, match="for the run from the layout"):
        _InstantOnly(context=None).execute(argument)
