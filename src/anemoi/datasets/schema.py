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
from typing import Annotated
from typing import Union

from anemoi.utils.dates import frequency_to_timedelta
from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import ValidationError
from pydantic import create_model
from pydantic import field_validator


class Deprecated:
    def __init__(self, name):
        self.name = name

    def __get_pydantic_core_schema__(self, *args, **kwargs):
        print("DEPRECATED FIELD", args, kwargs)
        return {"type": "str"}


class Interval(BaseModel):
    start: datetime.datetime = None
    end: datetime.datetime = None


class Datelist(BaseModel):
    values: list[datetime.datetime] = None


class Dates(Interval, Datelist):
    class Config:
        extra = "forbid"

    frequency: datetime.timedelta = None
    missing: list[datetime.datetime] = None

    @field_validator("frequency", mode="before")
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


class Step(BaseModel):
    pass


class FilteredStep(Step):
    dates: Interval


class Input(BaseModel):
    class Config:
        extra = "forbid"

    concat: list[FilteredStep] = None
    join: list[Step] = None

    accumulations: Step = None
    mars: Step = None
    constants: Step = None
    dates: Step = None
    netcdf: Step = None
    grib: Step = None


class Output(BaseModel):
    class Config:
        extra = "forbid"

    statistics_end: Union[datetime.datetime, int] = None

    chunking: dict = None
    dtype: str = "float32"
    flatten_grid: bool = True
    order_by: list[str] = None
    remapping: dict = None
    statistics: Union[dict, str] = None


class Build(BaseModel):
    class Config:
        extra = "forbid"

    group_by: Union[int, str] = "monthly"
    use_grib_paramid: bool = False
    variable_naming: str = None


class Common(BaseModel):

    pass


class Statistics(BaseModel):
    class Config:
        extra = "forbid"

    end: Union[datetime.datetime, int] = None
    allow_nans: list[str] = None


class Concat(BaseModel):
    concat: list


class Join(BaseModel):
    join: list


def _input_discriminator(input):

    if isinstance(input, dict):
        keys = sorted(input.keys())
        if len(keys) != 1:
            raise ValueError(f"Invalid input: {keys}")

        action = keys[0]
        if action in ("concat", "join"):
            return action

        return "other"

    print("DISCRIMINATOR", input)
    raise ValueError(f"Invalid input type: {type(input)}")


def other(*args, **kwargs):
    action = list(kwargs.keys())[0]
    name = action[0].upper() + action[1:].lower() + "Step"
    return create_model(name, **{action: (dict, ...)}, __base__=Step)


class Recipe(BaseModel):
    class Config:
        extra = "forbid"

    description: str = None
    name: str = None
    copyright: str = None
    licence: str = None

    dates: Dates

    # https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator
    input: Annotated[
        Union[
            Annotated[Concat, Tag("concat")],
            Annotated[Join, Tag("join")],
            Annotated[other, Tag("other")],
        ],
        Discriminator(_input_discriminator),
    ]

    output: Output = Output()
    build: Build = Build()
    statistics: Statistics = Statistics()
    common: Common = None
    sources: Common = None

    # Legacy fields
    dataset_status: str = Annotated[int, Field(deprecated="This is deprecated")]
    purpose: str = Annotated[int, Field(deprecated="This is deprecated")]
    attribution: str = None
    aliases: Union[Common, list] = None
    config_format_version: int = Annotated[int, Field(deprecated="This is deprecated")]
    flatten_grid: bool = True
    ensemble_dimension: int = 2
    status: str = Annotated[int, Field(deprecated="This is deprecated")]


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
