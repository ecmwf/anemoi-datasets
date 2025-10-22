# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import os
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
from earthkit.data import settings
from numpy.typing import NDArray


def cache_context(dirname: str) -> contextmanager:
    """Context manager for setting a temporary cache directory.

    Parameters
    ----------
    dirname : str
        The directory name for the cache.

    Returns
    -------
    contextmanager
        A context manager that sets the cache directory.
    """

    @contextmanager
    def no_cache_context():
        yield

    if dirname is None:
        return no_cache_context()

    os.makedirs(dirname, exist_ok=True)
    # return settings.temporary("cache-directory", dirname)
    return settings.temporary({"cache-policy": "user", "user-cache-directory": dirname})


def to_datetime_list(*args: Any, **kwargs: Any) -> list[datetime.datetime]:
    """Convert various date formats to a list of datetime objects.

    Parameters
    ----------
    *args : Any
        Positional arguments for date conversion.
    **kwargs : Any
        Keyword arguments for date conversion.

    Returns
    -------
    list[datetime.datetime]
        A list of datetime objects.
    """
    from earthkit.data.utils.dates import to_datetime_list as to_datetime_list_

    warnings.warn(
        "to_datetime_list() is deprecated. Call earthkit.data.utils.dates.to_datetime_list() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_datetime_list_(*args, **kwargs)


def to_datetime(*args: Any, **kwargs: Any) -> datetime.datetime:
    """Convert various date formats to a single datetime object.

    Parameters
    ----------
    *args : Any
        Positional arguments for date conversion.
    **kwargs : Any
        Keyword arguments for date conversion.

    Returns
    -------
    datetime.datetime
        A datetime object.
    """
    from earthkit.data.utils.dates import to_datetime as to_datetime_

    warnings.warn(
        "to_datetime() is deprecated. Call earthkit.data.utils.dates.to_datetime() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return to_datetime_(*args, **kwargs)


def normalize_and_check_dates(
    dates: list[datetime.datetime],
    start: datetime.datetime,
    end: datetime.datetime,
    frequency: datetime.timedelta,
    dtype: str = "datetime64[s]",
) -> NDArray[Any]:
    """Normalize and check a list of dates against a specified frequency.

    Parameters
    ----------
    dates : list[datetime.datetime]
        The list of dates to check.
    start : datetime.datetime
        The start date.
    end : datetime.datetime
        The end date.
    frequency : datetime.timedelta
        The frequency of the dates.
    dtype : str, optional
        The data type of the dates, by default "datetime64[s]".

    Returns
    -------
    NDArray[Any]
        An array of normalized dates.

    Raises
    ------
    ValueError
        If the final date size does not match the data shape.
    """
    dates = [d.hdate if hasattr(d, "hdate") else d for d in dates]

    assert isinstance(frequency, datetime.timedelta), frequency
    start = np.datetime64(start)
    end = np.datetime64(end)
    delta = np.timedelta64(frequency)

    res = []
    while start <= end:
        res.append(start)
        start += delta
    dates_ = np.array(res).astype(dtype)

    if len(dates_) != len(dates):
        raise ValueError(
            f"Final date size {len(dates_)} (from {dates_[0]} to {dates_[-1]}, "
            f"{frequency=}) does not match data shape {len(dates)} (from {dates[0]} to "
            f"{dates[-1]})."
        )

    for i, (d1, d2) in enumerate(zip(dates, dates_)):
        d1 = np.datetime64(d1)
        d2 = np.datetime64(d2)
        assert d1 == d2, (i, d1, d2)

    return dates_
