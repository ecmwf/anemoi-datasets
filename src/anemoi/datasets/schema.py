# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import json
from typing import Union

from anemoi.utils.dates import frequency_to_timedelta
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic import validator


class Dates(BaseModel):
    class Config:
        extra = "forbid"

    start: datetime.datetime
    end: datetime.datetime
    frequency: datetime.timedelta
    missing: list[datetime.datetime] = None

    @validator("frequency", pre=True)
    def parse_frequency(cls, value):
        return frequency_to_timedelta(value)

    group_by: Union[int, str] = None
    # @validator("group_by", pre=True, always=True)
    # def warn_deprecated_field(cls, value):
    #     if value is not None:
    #         warnings.warn(
    #             "The field 'group_by' is deprecated and will be removed in future versions.",
    #             DeprecationWarning,
    #             stacklevel=2,
    #         )
    #     return value


class Input(BaseModel):
    pass


class Output(BaseModel):
    pass


class Build(BaseModel):
    group_by: Union[int, str] = "monthly"


class Common(BaseModel):
    pass


class Statistics(BaseModel):
    pass


class Recipe(BaseModel):
    class Config:
        extra = "forbid"

    description: str = None
    name: str = None
    copyright: str = None
    licence: str = None

    dates: Dates
    input: Input
    output: Output = Output()
    build: Build = Build()
    statistics: Statistics = Statistics()
    common: Common = None

    # Legacy fields
    dataset_status: str = None
    purpose: str = None
    attribution: str = None
    aliases: Union[Common, list] = None
    config_format_version: int = None
    flatten_grid: bool = True
    ensemble_dimension: int = 2

    # @validator("dataset_status", pre=True, always=True)
    # def warn_deprecated_field(cls, value):
    #     if value is not None:
    #         warnings.warn(
    #             "The field 'dataset_status' is deprecated and will be removed in future versions.",
    #             DeprecationWarning,
    #             stacklevel=2,
    #         )
    #     return value


def validate(config):
    try:
        validated_data = Recipe(**config)
        print("Validation successful!")
        print(json.dumps(validated_data.dict(), default=str, indent=4))
        return True
    except ValidationError as e:
        print("Validation failed:")
        print(e)
        return False
