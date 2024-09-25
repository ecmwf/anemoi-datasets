# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import os
from contextlib import contextmanager

import numpy as np
from earthkit.data import settings


def cache_context(dirname):
    @contextmanager
    def no_cache_context():
        yield

    if dirname is None:
        return no_cache_context()

    os.makedirs(dirname, exist_ok=True)
    # return settings.temporary("cache-directory", dirname)
    return settings.temporary({"cache-policy": "user", "user-cache-directory": dirname})


def to_datetime_list(*args, **kwargs):
    from earthkit.data.utils.dates import to_datetime_list as to_datetime_list_

    return to_datetime_list_(*args, **kwargs)


def to_datetime(*args, **kwargs):
    from earthkit.data.utils.dates import to_datetime as to_datetime_

    return to_datetime_(*args, **kwargs)


def make_list_int(value):
    if isinstance(value, str):
        if "/" not in value:
            return [value]
        bits = value.split("/")
        if len(bits) == 3 and bits[1].lower() == "to":
            value = list(range(int(bits[0]), int(bits[2]) + 1, 1))

        elif len(bits) == 5 and bits[1].lower() == "to" and bits[3].lower() == "by":
            value = list(range(int(bits[0]), int(bits[2]) + int(bits[4]), int(bits[4])))

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return value
    if isinstance(value, int):
        return [value]

    raise ValueError(f"Cannot make list from {value}")


def normalize_and_check_dates(dates, start, end, frequency, dtype="datetime64[s]"):

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
