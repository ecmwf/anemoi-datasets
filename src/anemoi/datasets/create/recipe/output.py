# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from math import prod
from typing import Annotated
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Union

import numpy as np
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

    sanitise: bool = True
    """Whether to sanitise the metadata to remove sensitive information like paths, URLs..."""


class GriddedOutput(OutputBase):
    """Output configuration for gridded datasets."""

    _DEFAULT_GRID_SPLITS: ClassVar[int] = 4
    # Blosc and several other codecs use signed 32-bit buffer sizes.
    _MAX_CHUNK_BYTES: ClassVar[int] = 2**31 - 1
    _MIN_CHUNK_BYTES: ClassVar[int] = 2**25  # 32 MiB

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

        Unless an explicit ``values`` chunk size is configured and the date
        is larger than :attr:`_MIN_CHUNK_BYTES`, split the grid into four
        chunks. If the date is smaller than :attr:`_MIN_CHUNK_BYTES`, the grid
        is not split. If a chunk would exceed the codec buffer limit, the
        grid chunking is doubled until it fits.

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

        if "values" in coords and "values" not in self.chunking:
            grid_axis = list(coords).index("values")
            grid_size = len(coords["values"])
            splits = self._DEFAULT_GRID_SPLITS

            def set_grid_chunk_size() -> int:
                chunks[grid_axis] = max(1, (grid_size + splits - 1) // splits)
                return prod(chunks) * np.dtype(self.dtype).itemsize

            chunk_bytes = set_grid_chunk_size()
            while splits > 1 and chunk_bytes < self._MIN_CHUNK_BYTES:
                splits //= 2
                chunk_bytes = set_grid_chunk_size()

            while True:
                if chunk_bytes <= self._MAX_CHUNK_BYTES:
                    break
                if chunks[grid_axis] == 1:
                    raise ValueError(
                        f"A single-grid-point chunk requires {chunk_bytes:,} bytes, "
                        f"exceeding the {self._MAX_CHUNK_BYTES:,}-byte codec limit."
                    )
                splits *= 2
                chunk_bytes = set_grid_chunk_size()

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

    rows_per_chunk: int | None = None
    """The number of rows per chunk for tabular datasets. If None, the chunk size will be determined automatically based on the target fragment size and the row size."""

    auto_rows_per_chunk: str | None = "1d"
    """Iteration window (e.g. "1d", "6h") used by ``finalise --rows-per-chunk`` to choose ``rows_per_chunk`` automatically. When this is set and ``rows_per_chunk`` is None, finalise computes the chunk size that minimises read time for this window and re-chunks the data array. Set to null to disable and require an explicit ``rows_per_chunk``."""

    bytes_per_chunk: int = 64 * 1024 * 1024  # 64 MiB
    """The target size of each chunk in bytes for tabular datasets. This is used to determine the number of rows per chunk if rows_per_chunk is None."""


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
