# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field

LOG = logging.getLogger(__name__)


class Output(BaseModel):
    dtype: str = "float32"
    flatten_grid: bool = True

    order_by: list[str] = Field(default_factory=lambda: ["valid_datetime", "param_level", "number"])

    remapping: dict[str, Any] = Field(default_factory=lambda: {"param_level": "{param}_{levelist}"})
    chunking: dict[str, int] = Field(default_factory=lambda: {"dates": 1, "ensembles": 1})

    def _post_init(self, recipe: BaseModel) -> None:
        # We need to access `build`. This is for backward compatibility.
        # We need to find a better way to do that.
        if "param_level" in self.remapping:
            self.remapping["param_level"] = recipe.build.variable_naming

    def get_chunking(self, coords: dict) -> tuple:
        """Returns the chunking configuration based on coordinates.

        Parameters
        ----------
        coords : dict
            The coordinates dictionary.

        Returns
        -------
        tuple
            The chunking configuration.
        """
        user = self.chunking.copy()
        chunks = []
        for k, v in coords.items():
            if k in user:
                chunks.append(user.pop(k))
            else:
                chunks.append(len(v))
        if user:
            raise ValueError(
                f"Unused chunking keys from config: {list(user.keys())}, not in known keys : {list(coords.keys())}"
            )
        return tuple(chunks)
