# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Annotated

from pydantic import BaseModel
from pydantic import BeforeValidator

LOG = logging.getLogger(__name__)


def validate_variable_naming(value):
    NAMINGS = {
        "param": "{param}",
        "param_levelist": "{param}_{levelist}",
        "default": "{param}_{levelist}",
    }

    return NAMINGS.get(value, value)


def validate_mapping(value):
    assert False, validate_mapping


class Build(BaseModel):

    class Config:
        # arbitrary_types_allowed = True
        extra = "forbid"

    use_grib_paramid: bool = False
    allow_nans: bool = False
    variable_naming: Annotated[str, BeforeValidator(validate_variable_naming)] = validate_variable_naming("default")
