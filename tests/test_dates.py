# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import pytest

from anemoi.datasets.create.loaders import default_statistics_dates

_ = datetime.datetime


def date_list(start, end, step, missing=[]):
    step = datetime.timedelta(hours=step)
    start = datetime.datetime(*start)
    end = datetime.datetime(*end)
    missing = [datetime.datetime(m) for m in missing]
    current = start
    dates = []
    while current <= end:
        dates.append(current)
        current += step
    return dates


def default_start(*args, **kwargs):
    return default_statistics_dates(date_list(*args, **kwargs))[0]


def default_end(*args, **kwargs):
    return default_statistics_dates(date_list(*args, **kwargs))[1]


@pytest.mark.parametrize("y", [2000, 2001, 2002, 2003, 2004, 2005, 1900, 2100])
def test_default_statistics_dates(y):
    # >= 20 years
    assert default_end((y, 1, 1), (y + 19, 12, 23), 1) == datetime.datetime(y + 16, 12, 31, 23)
    assert default_end((y, 1, 1), (y + 20, 12, 23), 1) == datetime.datetime(y + 17, 12, 31, 23)
    assert default_end((y, 1, 1), (y + 19, 12, 23), 1) == datetime.datetime(y + 16, 12, 31, 23)

    # 19.51 years
    assert default_end((y, 1, 1), (y + 19, 7, 4), 1) == datetime.datetime(y + 16, 12, 31, 23)
    # 19.49 years
    assert default_end((y, 1, 1), (y + 19, 7, 2), 1) == datetime.datetime(y + 18, 12, 31, 23)

    # >= 10 years and < 20 years
    assert default_end((y, 1, 1), (y + 9, 12, 23), 1) == datetime.datetime(y + 8, 12, 31, 23)
    assert default_end((y, 1, 1), (y + 10, 12, 23), 1) == datetime.datetime(y + 9, 12, 31, 23)
    assert default_end((y, 1, 1), (y + 11, 12, 23), 1) == datetime.datetime(y + 10, 12, 31, 23)

    assert default_end((y, 1, 1), (y + 9, 12, 23), 6) == datetime.datetime(y + 8, 12, 31, 18)
    assert default_end((y, 1, 1), (y + 10, 12, 23), 6) == datetime.datetime(y + 9, 12, 31, 18)
    assert default_end((y, 1, 1), (y + 11, 12, 23), 6) == datetime.datetime(y + 10, 12, 31, 18)

    assert default_end((y, 1, 1), (y + 9, 12, 23), 12) == datetime.datetime(y + 8, 12, 31, 12)
    assert default_end((y, 1, 1), (y + 10, 12, 23), 12) == datetime.datetime(y + 9, 12, 31, 12)
    assert default_end((y, 1, 1), (y + 11, 12, 23), 12) == datetime.datetime(y + 10, 12, 31, 12)


def test_default_statistics_dates_80_percent():
    # < 10 years, keep 80% of the data
    assert default_end((2000, 1, 1), (2001, 12, 23), 1) == datetime.datetime(2001, 7, 31, 14)
    assert default_end((2000, 1, 1), (2002, 12, 23), 1) == datetime.datetime(2002, 5, 19, 14)


if __name__ == "__main__":
    test_default_statistics_dates()
