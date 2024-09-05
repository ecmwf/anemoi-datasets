# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from typing import Union

from pydantic import BaseModel

from .xarray import load_many


class Netcdf(BaseModel):
    class Config:
        extra = "forbid"

    path: Union[str, list[str]]
    param: Union[str, list[str]]


def execute(context, dates, path, *args, **kwargs):
    return load_many("📁", context, dates, path, *args, **kwargs)


schema = Netcdf
