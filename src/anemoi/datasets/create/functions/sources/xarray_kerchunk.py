# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from earthkit.data.core.fieldlist import MultiFieldList

from . import iterate_patterns
from .xarray import load_one


def load_many(emoji, context, dates, pattern, options, **kwargs):

    result = []
    options = options.copy() if options is not None else {}

    options.setdefault("engine", "zarr")
    options.setdefault("backend_kwargs", {})

    backend_kwargs = options["backend_kwargs"]
    backend_kwargs.setdefault("consolidated", False)
    backend_kwargs.setdefault("storage_options", {})

    storage_options = backend_kwargs["storage_options"]
    storage_options.setdefault("remote_protocol", "s3")
    storage_options.setdefault("remote_options", {"anon": True})

    for path, dates in iterate_patterns(pattern, dates, **kwargs):
        storage_options["fo"] = path

        result.append(load_one(emoji, context, dates, "reference://", options=options, **kwargs))

    return MultiFieldList(result)


def execute(context, dates, json, options=None, **kwargs):
    return load_many("🧱", context, dates, json, options, **kwargs)
