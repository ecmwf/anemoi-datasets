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
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag

LOG = logging.getLogger(__name__)


class OutputBase(BaseModel):
    """Base class for output configuration."""

    dtype: str = "float32"
    """The data type for the output dataset."""

    remapping: dict[str, Any] | None = Field(
        default=None,
        deprecated="'output.remapping' is deprecated. Please use 'build.remapping' instead.",
    )


class GriddedOutput(OutputBase):
    """Output configuration for gridded datasets."""

    format: Literal["gridded"] = "gridded"
    """The format of the dataset."""

    flatten_grid: bool = True
    """Whether to flatten the grid."""

    order_by: list[str] = Field(default_factory=lambda: ["valid_datetime", "param_level", "number"])
    """The order of dimensions in the output."""

    chunking: dict[str, int] = Field(default_factory=lambda: {"dates": 1, "ensembles": 1})
    """The chunking configuration for the output."""

    ensemble_dimension: int = 2
    """The ensemble dimension size."""

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


class TabularOutput(OutputBase):
    """Output configuration for tabular datasets."""

    format: Literal["tabular"] = "tabular"
    """The format of the dataset."""

    date_indexing: str = "bisect"
    """The date indexing method for tabular datasets. Options are "bisect", "btree"."""


def _output_discriminator(v: Any) -> str:
    """Discriminator function for Output union type."""
    if isinstance(v, dict):
        return v.get("layout", v.get("format", "gridded"))
    return getattr(v, "layout", getattr(v, "format", "gridded"))


# if layout is 'gridded', use GriddedOutput, if 'tabular', use TabularOutput
Output = Annotated[
    Union[
        Annotated[GriddedOutput, Tag("gridded")],
        Annotated[TabularOutput, Tag("tabular")],
    ],
    Discriminator(_output_discriminator),
]
