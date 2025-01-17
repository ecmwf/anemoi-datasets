# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from .xarray import load_many


def execute(context, dates, path, *args, **kwargs):
    options = {"engine": "fstd"}
    return load_many("🍁", context, dates, path, *args, options=options, **kwargs)