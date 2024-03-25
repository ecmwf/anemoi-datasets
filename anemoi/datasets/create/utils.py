# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import json
import os
from contextlib import contextmanager

import numpy as np
import yaml
from climetlab import settings
from climetlab.utils.humanize import seconds  # noqa: F401
from tqdm.auto import tqdm


def cache_context(dirname):
    @contextmanager
    def no_cache_context():
        yield

    if dirname is None:
        return no_cache_context()

    os.makedirs(dirname, exist_ok=True)
    return settings.temporary("cache-directory", dirname)


def bytes(n):
    """
    >>> bytes(4096)
    '4 KiB'
    >>> bytes(4000)
    '3.9 KiB'
    """
    if n < 0:
        sign = "-"
        n -= 0
    else:
        sign = ""

    u = ["", " KiB", " MiB", " GiB", " TiB", " PiB", " EiB", " ZiB", " YiB"]
    i = 0
    while n >= 1024:
        n /= 1024.0
        i += 1
    return "%s%g%s" % (sign, int(n * 10 + 0.5) / 10.0, u[i])


def to_datetime_list(*args, **kwargs):
    from climetlab.utils.dates import to_datetime_list as to_datetime_list_

    return to_datetime_list_(*args, **kwargs)


def to_datetime(*args, **kwargs):
    from climetlab.utils.dates import to_datetime as to_datetime_

    return to_datetime_(*args, **kwargs)


def load_json_or_yaml(path):
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)
        raise ValueError(f"Cannot read file {path}. Need json or yaml with appropriate extension.")


def compute_directory_sizes(path):
    if not os.path.isdir(path):
        return None

    size, n = 0, 0
    bar = progress_bar(iterable=os.walk(path), desc=f"Computing size of {path}")
    for dirpath, _, filenames in bar:
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            size += os.path.getsize(file_path)
            n += 1

    return dict(total_size=size, total_number_of_files=n)


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
    assert isinstance(frequency, int), frequency
    start = np.datetime64(start)
    end = np.datetime64(end)
    delta = np.timedelta64(frequency, "h")

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
        assert d1 == d2, (i, d1, d2)

    return dates_


def progress_bar(*, iterable=None, total=None, initial=0, desc=None):
    return tqdm(
        iterable=iterable,
        total=total,
        initial=initial,
        unit_scale=True,
        unit_divisor=1024,
        unit="B",
        disable=False,
        leave=False,
        desc=desc,
        # dynamic_ncols=True, # make this the default?
    )
