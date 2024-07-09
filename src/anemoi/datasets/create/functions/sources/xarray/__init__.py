# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import glob
import logging

from earthkit.data import from_source
from earthkit.data.core.fieldlist import MultiFieldList
from earthkit.data.utils.patterns import Pattern

from .fieldlist import XarrayFieldList

LOG = logging.getLogger(__name__)


def check(what, ds, paths, **kwargs):
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, {what}s={paths})")


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

        cnt = 0
        for p in glob.glob(path):
            yield p
            cnt += 1
        if cnt == 0:
            yield path


def load_one(context, dates, dataset, options, flavour=None, *args, **kwargs):
    import xarray as xr

    context.trace("🌐", dataset, options)

    if isinstance(dataset, str) and ".zarr" in dataset:
        data = xr.open_zarr(dataset, **options)
    else:
        data = xr.open_dataset(dataset, **options)

    fs = XarrayFieldList.from_xarray(data, flavour)
    return MultiFieldList([fs.sel(valid_datetime=date, **kwargs) for date in dates])


def load_many(emoji, what, context, dates, path, *args, **kwargs):
    given_paths = path if isinstance(path, list) else [path]

    dates = [d.isoformat() for d in dates]
    ds = from_source("empty")

    for path in given_paths:
        paths = Pattern(path, ignore_missing_keys=True).substitute(*args, date=dates, **kwargs)
        for path in _expand(paths):
            context.trace(emoji, what.upper(), path)
            s = load_one(context, dates, path, options={}, **kwargs)
            ds = ds + s

    # check(what, ds, given_paths, valid_datetime=dates, **kwargs)

    return ds


def execute(context, dates, url, *args, **kwargs):
    return load_many("🌐", "url", context, dates, url, *args, **kwargs)
