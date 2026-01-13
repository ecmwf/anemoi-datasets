# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import Field

if TYPE_CHECKING:
    from . import Recipe

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
    allow_nans: bool | list[str] = False
    """Allow NaN values in the dataset. Can be True, False, or a list of variable names."""
    variable_naming: Annotated[str, BeforeValidator(validate_variable_naming)] = validate_variable_naming("default")
    group_by: str | int = "monthly"
    additions: bool = False
    env: dict[str, str] = {}
    """Environment variables to set when creating the dataset."""
    remapping: dict[str, Any] = Field(default_factory=lambda: {"param_level": "{param}_{levelist}"})
    """Remapping configuration for the dataset."""

    def _post_init(self, recipe: Recipe) -> None:
        """Post-initialisation hook to handle legacy config options.

        Parameters
        ----------
        recipe : Recipe
            The parent recipe object.
        """
        # Use __dict__ to check deprecated fields without triggering Pydantic's
        # deprecation warnings. Only access the attribute if it has a non-None value.
        if recipe.__dict__.get("env") is not None:
            if self.env:
                raise ValueError("Cannot specify 'env' in both 'recipe' and 'build'. " "Please use 'build.env' only.")
            self.env = dict(recipe.env)
            recipe.env = None

        # Support legacy 'statistics.allow_nans'
        if recipe.statistics.__dict__.get("allow_nans") is not None:
            if self.allow_nans:
                raise ValueError(
                    "Cannot specify 'allow_nans' in both 'statistics' and 'build'. "
                    "Please use 'build.allow_nans' only."
                )
            self.allow_nans = recipe.statistics.allow_nans
            recipe.statistics.allow_nans = None

        # Support legacy 'output.remapping'
        if recipe.output.__dict__.get("remapping") is not None:
            if self.remapping:
                raise ValueError(
                    "Cannot specify 'remapping' in both 'output' and 'build'. " "Please use 'build.remapping' only."
                )
            self.remapping = dict(recipe.output.remapping)
            recipe.output.remapping = None

        # Apply variable_naming to remapping
        # This is for backward compatibility
        if "param_level" in self.remapping:
            self.remapping["param_level"] = self.variable_naming
