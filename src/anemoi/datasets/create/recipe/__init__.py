# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from typing import Annotated

import yaml
from anemoi.utils.config import DotDict
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import Field
from pydantic import model_validator

from .build import Build
from .output import GriddedOutput
from .output import Output
from .statistics import Statistics

LOG = logging.getLogger(__name__)


def validate_dotdict(v):
    if isinstance(v, dict):
        return DotDict(v)
    return v


DotDictField = Annotated[DotDict, BeforeValidator(validate_dotdict)]


class Recipe(BaseModel):

    @model_validator(mode="after")
    def _post_init(self) -> "Recipe":
        # We need to call _post_init on nested BaseModel members
        # So that they can do their own post-initialization
        # Once all members have been initialized
        for member in self.__dict__.values():
            if isinstance(member, BaseModel) and hasattr(member, "_post_init"):
                member._post_init(self)
        return self

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    description: str = "No description provided."
    licence: str = "unknown"
    attribution: str = "unknown"

    dates: DotDictField
    """The date configuration for the dataset."""

    input: DotDictField
    """The input data sources configuration."""

    data_sources: list[DotDictField] | DotDictField | None = None
    """The data sources configuration."""

    output: Output = Field(default_factory=GriddedOutput)
    """The output configuration."""

    build: Build = Build()
    """The build configuration."""

    statistics: Statistics = Statistics()

    env: dict[str, str] | None = Field(
        default=None,
        deprecated="Top-level 'env' is deprecated. Please use 'build.env' instead.",
    )


def loader_recipe_from_yaml(path: str) -> dict:
    """Load a dataset recipe from a YAML file.

    Parameters
    ----------
    path : str
        The path to the YAML file.

    Returns
    -------
    dict
        The dataset recipe.
    """
    with open(path) as f:
        recipe_yaml = f.read()
    recipe = yaml.safe_load(recipe_yaml)
    return Recipe(**recipe)


def loader_recipe_from_zarr(path: str) -> dict:
    """Load a dataset recipe from a Zarr store.

    Parameters
    ----------
    path : str
        The path to the Zarr store.

    Returns
    -------
    dict
        The dataset recipe.
    """
    import zarr

    z = zarr.open(path, mode="r")

    for name in ("_recipe", "recipe"):
        if name not in z.attrs:
            # return None
            LOG.error(f"No '{name}' found in Zarr store at {path}")
            continue

        recipe = z.attrs[name]
        recipe = recipe if isinstance(recipe, dict) else json.loads(recipe)
        return Recipe(**recipe)

    raise ValueError(f"No recipe found in Zarr store at {path}")
