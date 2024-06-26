# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import glob

from earthkit.data import from_source
from earthkit.data.utils.patterns import Pattern


def _expand(paths):
    for path in paths:
        if path.startswith("file://"):
            path = path[7:]

        if path.startswith("http://"):
            yield path
            continue

        if path.startswith("https://"):
            yield path
            continue

        for p in glob.glob(path):
            yield p


def check(what, ds, paths, **kwargs):
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, {what}s={paths})")


def load_netcdfs(emoji, what, context, dates, path, *args, **kwargs):
    given_paths = path if isinstance(path, list) else [path]

    dates = [d.isoformat() for d in dates]
    ds = from_source("empty")

    for path in given_paths:
        paths = Pattern(path, ignore_missing_keys=True).substitute(*args, date=dates, **kwargs)

        levels = kwargs.get("level", kwargs.get("levelist"))

        for path in _expand(paths):
            context.trace(emoji, what.upper(), path)
            s = from_source("opendap", path)
            s = s.sel(
                valid_datetime=dates,
                param=kwargs["param"],
                step=kwargs.get("step", 0),
            )
            if levels:
                s = s.sel(levelist=levels)
            ds = ds + s

    check(what, ds, given_paths, valid_datetime=dates, **kwargs)

    return ds


def execute(context, dates, path, *args, **kwargs):
    return load_netcdfs("📁", "path", context, dates, path, *args, **kwargs)
