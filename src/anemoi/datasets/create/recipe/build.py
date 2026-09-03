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
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import ConfigDict
from pydantic import Field
from pydantic_core import PydanticCustomError

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


def _tendencies_enabled(value: Any) -> bool:
    """Return whether a ``statistics.tendencies`` value enables tendency statistics.

    Parameters
    ----------
    value : Any
        The ``statistics.tendencies`` value (``bool``, ``None`` or a list of deltas).

    Returns
    -------
    bool
        ``True`` if tendency statistics are requested, ``False`` otherwise.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return bool(value)


def validate_mapping(value):
    assert False, validate_mapping


class Build(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    use_grib_paramid: bool = False
    allow_nans: bool | list[str] = False
    """Allow NaN values in the dataset. Can be True, False, or a list of variable names."""
    variable_naming: Annotated[str, BeforeValidator(validate_variable_naming)] = validate_variable_naming("default")
    group_by: str | int = "monthly"
    additions: bool | None = None
    env: dict[str, str] = {}

    # Below is for tabular datasete
    max_fragment_size: int = 1024 * 1024 * 1024  # 1 GiB
    validate_date_ranges: bool = False
    max_workers: int | None = None
    delete_files: bool = True  # Whether to delete temporary files as we go.
    write_zarr_buffer_size: int = 512 * 1024 * 1024

    # Parameters for the `finalise --rows-per-chunk` stage, which reports the chunk size (in rows)
    # that minimises read time when iterating a tabular dataset with a fixed time window.
    chunk_windows: list[str] = Field(default_factory=lambda: ["1h", "3h", "6h", "12h", "24h"])
    """Iteration window sizes to evaluate the optimal rows-per-chunk for, each treated independently."""
    chunk_alignment_offset: str | None = None
    """Offset of the window boundaries from the whole UTC hour (e.g. '30min', '3h'). None means windows end on a whole UTC hour."""
    chunk_compression_ratio: float = 0.5
    """Assumed on-disk bytes / raw bytes ratio, used to convert a chunk's row count into its on-disk read size."""
    fs_read_min_bytes: int = 64 * 1024 * 1024
    """Lower edge of the filesystem read sweet spot in bytes; reads smaller than this under-use the filesystem."""
    fs_read_max_bytes: int = 512 * 1024 * 1024
    """Upper edge of the filesystem read sweet spot in bytes; reads larger than this see throughput roll off."""
    fs_read_latency_seconds: float = 0.005
    """Fixed per-read cost in seconds (open + metadata + seek), paid once per chunk read."""
    fs_read_bandwidth_bytes_per_s: float = 2e9
    """Peak streaming bandwidth in bytes/second, achieved for reads inside the sweet spot."""

    """Environment variables to set when creating the dataset."""
    remapping: dict[str, Any] | None = Field(
        default=None,
        deprecated="'build.remapping' is deprecated. Please use 'build.variable_naming' instead.",
    )

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
                raise PydanticCustomError(
                    "conflicting_env",
                    "Cannot specify 'env' in both 'recipe' and 'build'. Please use 'build.env' only.",
                )
            self.env = dict(recipe.env)
            recipe.env = None

        # Support legacy 'statistics.allow_nans'
        if recipe.statistics.__dict__.get("allow_nans") is not None:
            if self.allow_nans:
                raise PydanticCustomError(
                    "conflicting_allow_nans",
                    "Cannot specify 'allow_nans' in both 'statistics' and 'build'. Please use 'build.allow_nans' only.",
                )
            self.allow_nans = recipe.statistics.allow_nans
            recipe.statistics.allow_nans = None

        # Support legacy 'output.remapping'
        if recipe.output.__dict__.get("remapping") is not None:
            if self.__dict__.get("remapping"):
                raise PydanticCustomError(
                    "conflicting_remapping",
                    "Cannot specify 'remapping' in both 'output' and 'build'. "
                    "Please use 'build.variable_naming' only.",
                )
            # Attribute access (not __dict__) so the deprecation warning fires.
            self.remapping = dict(recipe.output.remapping)
            recipe.output.remapping = None

        # Legacy recipes carried the field naming as an earthkit remapping
        # (e.g. {"param_level": "{param}_{levelist}"}). Substitute the
        # equivalent 'variable_naming' unless it was set explicitly.
        if self.__dict__.get("remapping"):
            from anemoi.transform.naming import variable_naming_from_remapping

            # Attribute access (not __dict__) so the deprecation warning fires.
            variable_naming = variable_naming_from_remapping(self.remapping)
            if variable_naming is not None:
                if self.variable_naming != validate_variable_naming("default"):
                    raise PydanticCustomError(
                        "conflicting_naming",
                        "Cannot specify both 'variable_naming' and a 'remapping' with a "
                        "'param_level' entry. Please use 'variable_naming' only.",
                    )
                self.variable_naming = validate_variable_naming(variable_naming)

        # Reconcile the deprecated 'build.additions' flag with the canonical
        # 'statistics.tendencies' option. The rest of the codebase relies
        # solely on 'statistics.tendencies'.
        self._resolve_tendencies(recipe)

    def _resolve_tendencies(self, recipe: Recipe) -> None:
        """Reconcile the deprecated ``build.additions`` flag with ``statistics.tendencies``.

        ``statistics.tendencies`` is the canonical option used throughout the
        codebase. ``build.additions`` is retained for backward compatibility and
        is resolved here:

        * If ``statistics.tendencies`` is provided, it is honoured.
        * If only ``build.additions`` is provided, its value is copied onto
          ``statistics.tendencies`` and a deprecation warning is issued.
        * If both are provided, any value that is ``None`` is ignored. If the
          remaining values agree, that value is used. If they disagree, an
          error is raised.

        Parameters
        ----------
        recipe : Recipe
            The parent recipe object.
        """
        statistics = recipe.statistics

        # ``None`` means "not provided" for both options (the round-trip through
        # ``model_dump_json`` re-injects defaults, so we rely on the value rather
        # than ``model_fields_set``).
        if self.additions is None:
            # Nothing to reconcile: 'statistics.tendencies' (set or default) wins.
            return

        if statistics.tendencies is not None:
            if _tendencies_enabled(statistics.tendencies) != bool(self.additions):
                raise PydanticCustomError(
                    "conflicting_tendencies",
                    "Conflicting configuration: 'build.additions' ({additions}) and "
                    "'statistics.tendencies' ({tendencies}) disagree. "
                    "Please use 'statistics.tendencies' only.",
                    {"additions": self.additions, "tendencies": statistics.tendencies},
                )
            # They agree: keep the (richer) 'statistics.tendencies' value.
            return

        # Only the deprecated 'build.additions' was provided.
        LOG.warning("'build.additions' is deprecated; please use 'statistics.tendencies' instead.")
        statistics.tendencies = self.additions
