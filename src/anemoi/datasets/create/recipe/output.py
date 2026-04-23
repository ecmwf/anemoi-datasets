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

    @property
    def layout(self) -> str:
        return self.format

    order_by: list[str] | None = Field(
        default=None,
        deprecated=(
            "'output.order_by' is deprecated and no longer read from the recipe. "
            "The cube ordering is hard-coded to "
            "['valid_datetime', 'param_level', 'number']. Remove this key from "
            "the recipe."
        ),
    )
    """Deprecated.  Kept temporarily so existing recipes keep parsing, but
    it is no longer honoured: the cube ordering is hard-coded in
    :class:`SimpleGriddedContext`.  If present, it must match the fixed
    default value; any other value raises an error (see
    :meth:`Output._post_init`-style validation in :class:`Recipe`)."""

    chunking: dict[str, int] = Field(default_factory=lambda: {"dates": 1, "ensembles": 1})
    """The chunking configuration for the output."""

    ensemble_dimension: int = 2
    """The ensemble dimension size."""

    # Fixed value that the deprecated ``order_by`` field must match, if set.
    # Kept in sync with ``SimpleGriddedContext.order_by``.
    _FIXED_ORDER_BY = ["valid_datetime", "param_level", "number"]

    def _post_init(self, recipe: Any) -> None:
        """Validate the deprecated ``order_by`` field.

        Accept the value only if it equals the hard-coded default; any other
        value is rejected.  Emit a ``DeprecationWarning`` when the user has
        set the field in the recipe (even to the default value).
        """
        if "order_by" not in self.model_fields_set:
            return

        import warnings

        user = self.__dict__.get("order_by")
        if user is not None and list(user) != self._FIXED_ORDER_BY:
            raise ValueError(
                "'output.order_by' is deprecated and the cube ordering is now "
                f"hard-coded to {self._FIXED_ORDER_BY}. Got {list(user)!r}."
            )
        warnings.warn(
            "'output.order_by' is deprecated and no longer read from the "
            "recipe. The cube ordering is hard-coded to "
            f"{self._FIXED_ORDER_BY}. Remove this key from the recipe.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Drop the user-supplied value so it is not persisted in metadata.
        self.__dict__["order_by"] = None

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

    @property
    def layout(self) -> str:
        return self.format

    date_indexing: str = "bisect"
    """The date indexing method for tabular datasets. Options are "bisect", "btree"."""

class TrajectoriesOutput(OutputBase):
    """Output configuration for trajectory datasets.

    Unlike :class:`GriddedOutput`, this class has no user-configurable
    ``order_by``: the trajectory cube ordering is an internal detail
    tightly coupled to the composite ``traj_point`` remapping key injected
    by :class:`TrajectoryGriddedContext`, and per-field placement in
    :meth:`TrajectoryGriddedCreator.load_result` reads ``date/time/step``
    from field metadata directly.
    """

    layout: Literal["trajectories"] = "trajectories"

    chunking: dict[str, int] = Field(default_factory=lambda: {"base_dates": 1, "steps": 1, "ensembles": 1})
    """Chunking configuration for the 5-D output array (base_dates, variables, ensembles, steps, cells)."""

    def get_chunking(self, coords: dict) -> tuple:
        """Return chunking tuple for the 5-D Zarr array.

        Parameters
        ----------
        coords : dict
            Coordinate arrays keyed by dimension name.

        Returns
        -------
        tuple
            Chunk sizes in dimension order.
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
                f"Unused chunking keys from config: {list(user.keys())}, not in known keys: {list(coords.keys())}"
            )
        return tuple(chunks)


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
        Annotated[TrajectoriesOutput, Tag("trajectories")],
    ],
    Discriminator(_output_discriminator),
]
