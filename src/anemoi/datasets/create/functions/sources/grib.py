# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from climetlab import load_source
from climetlab.utils.patterns import Pattern


def check(ds, paths, **kwargs):
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, paths={paths})")


def execute(context, dates, path, *args, **kwargs):
    given_paths = path if isinstance(path, list) else [path]

    ds = load_source("empty")
    dates = [d.isoformat() for d in dates]

    for path in given_paths:
        paths = Pattern(path, ignore_missing_keys=True).substitute(*args, date=dates, **kwargs)

        for name in ("grid", "area", "rotation", "frame", "resol", "bitmap"):
            if name in kwargs:
                raise ValueError(f"MARS interpolation parameter '{name}' not supported")

        for path in paths:
            context.trace("📁", "PATH", path)
            s = load_source("file", path)
            s = s.sel(valid_datetime=dates, **kwargs)
            ds = ds + s

    if kwargs:
        check(ds, given_paths, valid_datetime=dates, **kwargs)

    return ds
