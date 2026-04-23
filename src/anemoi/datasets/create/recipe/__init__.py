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
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from .build import Build
from .output import GriddedOutput
from .output import Output
from .output import TrajectoriesOutput
from .statistics import Statistics

LOG = logging.getLogger(__name__)


def validate_dotdict(v):
    if isinstance(v, dict):
        return DotDict(v)
    return v


DotDictField = Annotated[DotDict, BeforeValidator(validate_dotdict)]


class Recipe(BaseModel):

    @model_validator(mode="after")
    def _check_steps(self) -> "Recipe":
        is_traj = isinstance(self.output, TrajectoriesOutput)
        if is_traj and self.steps is None:
            raise ValueError("'steps' is required when output layout is 'trajectories'")
        if is_traj:
            if self.base_dates is None:
                raise ValueError(
                    "'base_dates' is required when output layout is 'trajectories' "
                    "(use 'base_dates:' instead of 'dates:')"
                )
            if self.dates is not None:
                raise ValueError(
                    "'dates' is not accepted for the 'trajectories' layout; "
                    "use 'base_dates:' instead"
                )
        else:
            if self.base_dates is not None:
                raise ValueError(
                    "'base_dates' is only accepted for the 'trajectories' layout"
                )
            if self.dates is None:
                raise ValueError("'dates' is required")
        return self

    @model_validator(mode="after")
    def _post_init(self) -> "Recipe":
        # We need to call _post_init on nested BaseModel members
        # So that they can do their own post-initialization
        # Once all members have been initialized
        for member in self.__dict__.values():
            if isinstance(member, BaseModel) and hasattr(member, "_post_init"):
                member._post_init(self)
        return self

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    description: str = "No description provided."
    licence: str = "unknown"
    attribution: str = "unknown"

    dates: DotDictField | None = None
    """The date configuration for gridded and tabular datasets.  Mutually
    exclusive with ``base_dates`` (which is the trajectories equivalent)."""

    base_dates: DotDictField | None = None
    """The base-date (forecast initialisation time) configuration for the
    ``trajectories`` layout.  Mutually exclusive with ``dates``."""

    input: DotDictField
    """The input data sources configuration."""

    data_sources: list[DotDictField] | DotDictField | None = None
    """The data sources configuration."""

    output: Output = Field(default_factory=GriddedOutput)
    """The output configuration."""

    build: Build = Build()
    """The build configuration."""
    additions: DotDictField | None = Field(
        default=None,
        deprecated="Top-level 'additions' is deprecated. Use 'statistics.tendencies' instead.",
    )

    statistics: Statistics = Statistics()

    steps: DotDictField | None = None
    """The steps configuration for trajectory datasets (start, end, frequency)."""

    env: dict[str, str] | None = Field(
        default=None,
        deprecated="Top-level 'env' is deprecated. Please use 'build.env' instead.",
    )

    def strip_unknown_keys(self, data: dict) -> dict:
        assert isinstance(data, dict)
        defaults = Recipe(dates={}, input={}).model_dump()
        result = {key: data[key] for key in defaults.keys()}
        # Trajectory-only keys are omitted when unused, so gridded/tabular
        # recipes keep the same metadata shape they had before these fields
        # existed.
        for key in ("base_dates", "steps"):
            if result.get(key) is None:
                result.pop(key, None)
        return result


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
    LOG.info(f"Loading recipe from YAML file at {path}")

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

    LOG.info(f"Loading recipe from Zarr store at {path}")

    z = zarr.open(path, mode="r")

    for name in ("_recipe", "recipe"):
        if name not in z.attrs:
            continue

        recipe = z.attrs[name]
        recipe = recipe if isinstance(recipe, dict) else json.loads(recipe)
        return Recipe(**recipe)

    raise ValueError(f"No recipe found in Zarr store at {path}")
