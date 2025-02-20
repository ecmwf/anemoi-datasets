# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime

import numpy as np
import pytest

from anemoi.datasets.create.statistics import default_statistics_dates

_ = datetime.datetime


def date_list(start: tuple, end: tuple, step: int, missing: list = [], as_numpy: bool = False) -> list:
    """
    Generate a list of dates from start to end with a given step.

    Args:
        start (tuple): Start date as (year, month, day).
        end (tuple): End date as (year, month, day).
        step (int): Step in hours.
        missing (list): List of missing dates.
        as_numpy (bool): Whether to return dates as numpy.datetime64.

    Returns:
        list: List of dates.
    """
    step = datetime.timedelta(hours=step)
    start = datetime.datetime(*start)
    end = datetime.datetime(*end)
    missing = [datetime.datetime(m) for m in missing]
    current = start
    dates = []
    while current <= end:
        dates.append(current)
        current += step
    if as_numpy:
        dates = [np.datetime64(d) for d in dates]
    return dates


def default_start(*args, **kwargs) -> datetime.datetime:
    """
    Get the default start date for statistics.

    Returns:
        datetime.datetime: Default start date.
    """
    return default_statistics_dates(date_list(*args, **kwargs))[0]


def default_end(*args, **kwargs) -> datetime.datetime:
    """
    Get the default end date for statistics.

    Returns:
        datetime.datetime: Default end date.
    """
    return default_statistics_dates(date_list(*args, **kwargs))[1]


@pytest.mark.parametrize("y", [2000, 2001, 2002, 2003, 2004, 2005, 1900, 2100])
@pytest.mark.parametrize("as_numpy", [True, False])
def test_default_statistics_dates(y: int, as_numpy: bool) -> None:
    """
    Test the default_statistics_dates function with various inputs.

    Args:
        y (int): Year.
        as_numpy (bool): Whether to use numpy.datetime64.
    """
    assert default_start((y, 1, 1), (y + 19, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(y, 1, 1, 0)

    # >= 20 years
    assert default_end((y, 1, 1), (y + 19, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(y + 16, 12, 31, 23)
    assert default_end((y, 1, 1), (y + 20, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(y + 17, 12, 31, 23)
    assert default_end((y, 1, 1), (y + 19, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(y + 16, 12, 31, 23)

    # 19.51 years
    assert default_end((y, 1, 1), (y + 19, 7, 4), 1, as_numpy=as_numpy) == datetime.datetime(y + 16, 12, 31, 23)
    # 19.49 years
    assert default_end((y, 1, 1), (y + 19, 7, 2), 1, as_numpy=as_numpy) == datetime.datetime(y + 18, 12, 31, 23)

    # >= 10 years and < 20 years
    assert default_end((y, 1, 1), (y + 9, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(y + 8, 12, 31, 23)
    assert default_end((y, 1, 1), (y + 10, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(y + 9, 12, 31, 23)
    assert default_end((y, 1, 1), (y + 11, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(y + 10, 12, 31, 23)

    assert default_end((y, 1, 1), (y + 9, 12, 23), 6, as_numpy=as_numpy) == datetime.datetime(y + 8, 12, 31, 18)
    assert default_end((y, 1, 1), (y + 10, 12, 23), 6, as_numpy=as_numpy) == datetime.datetime(y + 9, 12, 31, 18)
    assert default_end((y, 1, 1), (y + 11, 12, 23), 6, as_numpy=as_numpy) == datetime.datetime(y + 10, 12, 31, 18)

    assert default_end((y, 1, 1), (y + 9, 12, 23), 12, as_numpy=as_numpy) == datetime.datetime(y + 8, 12, 31, 12)
    assert default_end((y, 1, 1), (y + 10, 12, 23), 12, as_numpy=as_numpy) == datetime.datetime(y + 9, 12, 31, 12)
    assert default_end((y, 1, 1), (y + 11, 12, 23), 12, as_numpy=as_numpy) == datetime.datetime(y + 10, 12, 31, 12)


@pytest.mark.parametrize("as_numpy", [True, False])
def test_default_statistics_dates_80_percent(as_numpy: bool) -> None:
    """
    Test the default_statistics_dates function for datasets less than 10 years.

    Args:
        as_numpy (bool): Whether to use numpy.datetime64.
    """
    # < 10 years, keep 80% of the data
    assert default_end((2000, 1, 1), (2001, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(2001, 7, 31, 14)
    assert default_end((2000, 1, 1), (2002, 12, 23), 1, as_numpy=as_numpy) == datetime.datetime(2002, 5, 19, 14)


if __name__ == "__main__":
    test_default_statistics_dates(2000, as_numpy=True)
