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

from .build import Build
from .output import Output
from .statistics import Statistics

LOG = logging.getLogger(__name__)


def validate_dotdict(v):
    if isinstance(v, dict):
        return DotDict(v)
    return v


DotDictField = Annotated[DotDict, BeforeValidator(validate_dotdict)]


class Recipe(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    description: str = "No description provided."
    licence: str = "unknown"
    attribution: str = "unknown"

    format: str = "gridded"
    """The format of the dataset. Options are "gridded", "tabular", etc."""

    dates: DotDictField
    """The date configuration for the dataset."""

    input: DotDictField
    """The input data sources configuration."""

    data_sources: list[DotDictField] | None = None
    """The data sources configuration."""

    date_indexing: str = "bisect"
    """The date indexing method for tabular datasets. Options are "bisect", "btree" """

    output: Output = Output()
    """The output  configuration."""

    build: Build = Build()
    """The build configuration."""

    statistics: Statistics = Statistics()

    env: dict[str, str] = Field(default_factory=dict)
    """Environment variables to set when creating the dataset."""


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
    if "_recipe" not in z.attrs:
        raise ValueError(f"No recipe found in Zarr store at {path}")

    recipe = json.loads(z.attrs["_recipe"])
    return Recipe(**recipe)
