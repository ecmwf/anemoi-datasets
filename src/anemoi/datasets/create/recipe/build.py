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
import re
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

# Mapping from legacy bare template keys to earthkit 1.0 component paths.
# - ``parameter.variable`` survives ``field.set()`` and
#   ``new_field_from_latitudes_longitudes`` wrapping.
# - ``vertical.level`` also survives all wrapping; it returns 0 for surface
#   fields.  A ``Patch`` on ``param_level`` strips the resulting ``_0`` suffix
#   for surface variables (see ``result.py`` and ``context.py``).
_LEGACY_KEY_MAP = {
    "param": "parameter.variable",
    "level": "vertical.level",
    "levelist": "vertical.level",
}

# Pattern that matches bare {key} or {key:type} template variables but leaves
# already-prefixed {component.key} forms (e.g. {parameter.variable}) untouched.
# Group 1: key name (no dots → bare key).  Group 2: optional eccodes type
# qualifier (e.g. ":d" for double, ":l" for long).
_BARE_TEMPLATE_VAR = re.compile(r"\{(\w+)(:\w+)?\}")


def _to_earthkit10_template(template: str) -> str:
    """Translate bare ``{key}`` → earthkit 1.0 path ``{component.key}``.

    Known legacy keys (``param``, ``level``, ``levelist``) are mapped to their
    earthkit 1.0 equivalents (``parameter.variable``, ``vertical.level``).
    Already-prefixed forms such as ``{parameter.variable}`` are left unchanged
    (idempotent).  Unknown bare keys fall back to ``{metadata.key}`` so that
    custom remapping entries still work.

    Eccodes type qualifiers (e.g. ``{level:d}`` for double) are preserved:
    the base key is mapped and the qualifier is appended.  When a type
    qualifier is present, the mapping always targets ``metadata.key:type``
    (not a component path) because component paths like ``vertical.level:d``
    are not supported.
    """

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        type_qual = m.group(2) or ""  # e.g. ":d" or ""

        if type_qual:
            # Type-qualified keys must use metadata.key:type — component
            # paths (e.g. vertical.level:d) do not support eccodes types.
            return "{metadata." + key + type_qual + "}"
        return "{" + _LEGACY_KEY_MAP.get(key, f"metadata.{key}") + "}"

    return _BARE_TEMPLATE_VAR.sub(_replace, template)


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
    max_fragment_size: int = 268435456  # 256 MiB
    validate_date_ranges: bool = True
    max_workers: int | None = None
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
            if self.remapping:
                raise PydanticCustomError(
                    "conflicting_remapping",
                    "Cannot specify 'remapping' in both 'output' and 'build'. Please use 'build.remapping' only.",
                )
            self.remapping = dict(recipe.output.remapping)
            recipe.output.remapping = None

        # Reconcile the deprecated 'build.additions' flag with the canonical
        # 'statistics.tendencies' option. The rest of the codebase relies
        # solely on 'statistics.tendencies'.
        self._resolve_tendencies(recipe)

        # Apply variable_naming to remapping, translating bare {key} template
        # variables to {metadata.key} form required by earthkit 1.0's _get_single.
        # The variable_naming attribute retains the legacy {param}_{levelist} form
        # for use with field.metadata() in trajectory loading (which still accepts
        # bare keys); only the remapping dict used with to_cube() needs the prefix.
        if "param_level" in self.remapping:
            self.remapping["param_level"] = _to_earthkit10_template(self.variable_naming)

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
