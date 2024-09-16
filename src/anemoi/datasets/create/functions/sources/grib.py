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
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.data.utils.patterns import Pattern


class AddGrid:

    def __init__(self, field, latitudes, longitudes):
        self._field = field
        self._latitudes = latitudes
        self._longitudes = longitudes

    def __getattr__(self, name):
        return getattr(self._field, name)

    def __repr__(self) -> str:
        return repr(self._field)

    def grid_points(self):
        return self._latitudes, self._longitudes


def check(ds, paths, **kwargs):
    count = 1
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            count *= len(v)

    if len(ds) != count:
        raise ValueError(f"Expected {count} fields, got {len(ds)} (kwargs={kwargs}, paths={paths})")


def _expand(paths):
    for path in paths:
        cnt = 0
        for p in glob.glob(path):
            yield p
            cnt += 1
        if cnt == 0:
            yield path


def execute(context, dates, path, latitudes=None, longitudes=None, *args, **kwargs):
    given_paths = path if isinstance(path, list) else [path]

    if latitudes is not None and longitudes is not None:
        context.info(f"Using latitudes and longitudes from {latitudes} and {longitudes}")
        latitudes = from_source("file", latitudes)[0].to_numpy(flatten=True)
        longitudes = from_source("file", longitudes)[0].to_numpy(flatten=True)
        context.info(f"Latitudes: {len(latitudes)}, Longitudes: {len(longitudes)}")
        assert len(latitudes) == len(longitudes)

    ds = from_source("empty")
    dates = [d.isoformat() for d in dates]

    for path in given_paths:
        paths = Pattern(path, ignore_missing_keys=True).substitute(*args, date=dates, **kwargs)

        for name in ("grid", "area", "rotation", "frame", "resol", "bitmap"):
            if name in kwargs:
                raise ValueError(f"MARS interpolation parameter '{name}' not supported")

        for path in _expand(paths):
            context.trace("📁", "PATH", path)
            s = from_source("file", path)
            s = s.sel(valid_datetime=dates, **kwargs)
            ds = ds + s

    if kwargs:
        check(ds, given_paths, valid_datetime=dates, **kwargs)

    if latitudes is not None and longitudes is not None:
        ds = FieldArray([AddGrid(_, latitudes, longitudes) for _ in ds])

    return ds
