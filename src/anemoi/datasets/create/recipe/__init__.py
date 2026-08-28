# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from typing import TYPE_CHECKING
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from .action import Action
from .build import Build
from .dates import BaseDates
from .dates import Dates
from .dates import Steps
from .output import GriddedOutput
from .output import Output
from .output import TabularOutput
from .output import TrajectoriesOutput
from .statistics import Statistics

if TYPE_CHECKING:
    from anemoi.datasets.dates.groups import Groups
    from anemoi.datasets.dates.groups import TrajectoryGroups

LOG = logging.getLogger(__name__)


def _iter_accumulate_blocks(node: Any):
    """Yield every accumulate-block payload found in an input/data_sources tree.

    Payloads are yielded as ``AccumulateSchema`` instances: validated Action
    models carry them directly; raw dicts (e.g. under ``concat``) are
    validated on the fly.
    """
    from anemoi.datasets.create.sources.accumulate.description import AccumulateSchema

    if node is None:
        return
    if isinstance(node, AccumulateSchema):
        yield node
        return
    if isinstance(node, BaseModel):
        for name in type(node).model_fields:
            yield from _iter_accumulate_blocks(getattr(node, name))
        if node.model_extra:
            for value in node.model_extra.values():
                yield from _iter_accumulate_blocks(value)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "accumulate" and isinstance(value, dict):
                yield AccumulateSchema.model_validate(value)
            else:
                yield from _iter_accumulate_blocks(value)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_accumulate_blocks(item)


#: The recipe spellings of the time-reduction sources (see
#: ``create/sources/reduce_support/``).  They share one schema, so one iterator finds
#: every block whatever the verb.
REDUCE_NAMES = ("average", "minimum", "maximum")


def _iter_reduce_blocks(node: Any):
    """Yield every ``average``/``minimum``/``maximum`` payload in an input tree.

    Payloads are yielded as ``(name, ReduceSchema)`` pairs: validated Action
    models carry the schema directly; raw dicts (e.g. under ``concat``) are
    validated on the fly.
    """
    from anemoi.datasets.create.sources.reduce_support import ReduceSchema

    if node is None:
        return
    if isinstance(node, BaseModel):
        for name in type(node).model_fields:
            value = getattr(node, name)
            if name in REDUCE_NAMES and isinstance(value, ReduceSchema):
                yield name, value
            else:
                yield from _iter_reduce_blocks(value)
        if node.model_extra:
            for value in node.model_extra.values():
                yield from _iter_reduce_blocks(value)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            if key in REDUCE_NAMES and isinstance(value, dict):
                yield key, ReduceSchema.model_validate(value)
            else:
                yield from _iter_reduce_blocks(value)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _iter_reduce_blocks(item)


class Recipe(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _migrate_dates_group_by(cls, data: Any) -> Any:
        """Move the legacy ``dates.group_by`` key to ``build.group_by``.

        ``group_by`` controls how dates are grouped into build parts and belongs
        to the ``build`` section.  Older recipes placed it under ``dates``, where
        the ``Dates`` model silently ignores it (so it would fall back to the
        ``build.group_by`` default).  Honour it here instead, mirroring
        :func:`anemoi.datasets.commands.recipe.migrate.migrate_group_by`, so
        every recipe-loading path (analyse, init, load) behaves consistently.
        """
        if not isinstance(data, dict):
            return data
        dates = data.get("dates")
        if isinstance(dates, dict) and "group_by" in dates:
            # Copy rather than mutate the caller's dicts in place.
            dates = dict(dates)
            group_by = dates.pop("group_by")
            build = dict(data.get("build") or {})
            build.setdefault("group_by", group_by)
            data = {**data, "dates": dates, "build": build}
            LOG.warning(
                "'dates.group_by' is deprecated; use 'build.group_by'. " "Moved group_by=%r to build.group_by.",
                group_by,
            )
        return data

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
                raise ValueError("'dates' is not accepted for the 'trajectories' layout; " "use 'base_dates:' instead")
        else:
            if self.base_dates is not None:
                raise ValueError("'base_dates' is only accepted for the 'trajectories' layout")
            if self.dates is None:
                raise ValueError("'dates' is required")
        return self

    @model_validator(mode="after")
    def _check_accumulate(self) -> "Recipe":
        """Layout-dependent rules for accumulate blocks.

        ``from:`` describes the *subsource* (what the data is); the output
        layout decides the *output*.  They are orthogonal, so most ``from:``
        descriptions are accepted in both layouts — but two rules still need
        the layout the per-block schema cannot see: the ``from-layout``
        sentinel is meaningful only under a trajectory layout, and a bare
        ``from:`` (base-less, validity-time source data) must be checked for
        compatibility with the requested period.
        """
        from anemoi.utils.dates import frequency_to_string

        from ..sources.accumulate.description import FromBare
        from ..sources.accumulate.description import FromTrajectories
        from ..sources.accumulate.description import check_valid_time_source

        is_traj = isinstance(self.output, TrajectoriesOutput)
        blocks = list(_iter_accumulate_blocks(self.input))
        blocks += list(_iter_accumulate_blocks(self.data_sources))

        for block in blocks:
            from_ = block.from_

            # `from-layout` inherits the run grid from the output layout, so it
            # is meaningful only under a trajectory layout — reject it elsewhere.
            if isinstance(from_, FromTrajectories) and from_.is_layout_grid and not is_traj:
                raise ValueError(
                    "accumulate: 'from.base_dates: from-layout' / 'from.steps: from-layout' is only "
                    "valid in a 'layout: trajectories' recipe — the sentinel inherits the run grid "
                    "from the output layout, which only that layout imposes"
                )

            # A bare `from:` (accumulation only) describes base-less,
            # validity-time-indexed source data in *any* layout: in a
            # trajectory recipe the field valid at base_date + step is
            # relabelled onto that row; elsewhere it is summed by validity time.
            # Either way the accumulation must be a fixed duration (a base-anchored
            # source rejects the base-less request later, at dispatch).
            if isinstance(from_, FromBare):
                check_valid_time_source(from_, period=block.period)

            if is_traj and self.steps is not None and self.steps.start < block.period:
                # The output layout imposes a base per row; an output window
                # must not straddle the basetime.
                raise ValueError(
                    f"accumulate: 'steps.start' ({frequency_to_string(self.steps.start)}) must "
                    f"be >= the accumulate 'period' ({frequency_to_string(block.period)}) — "
                    "output windows must not straddle the basetime"
                )
        return self

    @model_validator(mode="after")
    def _check_reduce(self) -> "Recipe":
        """Layout-dependent rules for the time-reduction blocks.

        Two rules need the output layout, which the per-block schema cannot
        see: a run-anchored ``from:`` inherits its run from a trajectory
        layout, and its window must fit inside that run.  Both apply *only* to
        the run-anchored shape — a base-less ``from:`` reads an analysis
        archive, where data before the basetime exists and is the same
        quantity, so neither rule holds there.
        """
        from anemoi.utils.dates import frequency_to_string

        is_traj = isinstance(self.output, TrajectoriesOutput)
        blocks = list(_iter_reduce_blocks(self.input))
        blocks += list(_iter_reduce_blocks(self.data_sources))

        for name, block in blocks:
            if not block.is_run_anchored:
                continue

            if not is_traj:
                raise ValueError(
                    f"{name}: 'from.base_dates: true' is only valid in a "
                    "'layout: trajectories' recipe — the sentinel inherits the run from the "
                    "output layout, which only that layout imposes. Describe base-less source "
                    "data with 'from: {frequency: ...}' instead"
                )

            if self.steps is not None and self.steps.start < block.period:
                raise ValueError(
                    f"{name}: 'steps.start' ({frequency_to_string(self.steps.start)}) must be "
                    f">= 'period' ({frequency_to_string(block.period)}) — the samples come "
                    "from the run the layout imposes, so an output window must not straddle "
                    "the basetime"
                )
        return self

    @model_validator(mode="after")
    def _post_init(self) -> "Recipe":
        # We need to call _post_init on nested BaseModel members
        # So that they can do their own post-initialization
        # Once all members have been initialized
        for member in self.__dict__.values():
            if isinstance(member, BaseModel) and hasattr(member, "_post_init"):
                member._post_init(self)

        if isinstance(self.output, TrajectoriesOutput) and "group_by" not in self.build.model_fields_set:
            self.build.group_by = 1

        return self

    description: str = "No description provided."
    licence: str = "unknown"
    attribution: str = "unknown"

    dates: Dates | None = None
    """The date configuration for gridded and tabular datasets.  Mutually
    exclusive with ``base_dates`` (which is the trajectories equivalent)."""

    base_dates: BaseDates | None = None
    """The base-date (forecast initialisation time) configuration for the
    ``trajectories`` layout.  Mutually exclusive with ``dates``."""

    input: Action | None = None
    """The input data sources configuration."""

    data_sources: dict[str, Action] | list[Action] | None = None
    """The data sources configuration."""

    output: Output = Field(default_factory=GriddedOutput)
    """The output configuration."""

    build: Build = Build()
    """The build configuration."""

    additions: dict | None = Field(
        default=None,
        deprecated="Top-level 'additions' is deprecated. Use 'statistics.tendencies' instead.",
    )

    statistics: Statistics = Statistics()

    steps: Steps | None = None
    """The steps configuration for trajectory datasets (start, end, frequency)."""

    env: dict[str, str] | None = Field(
        default=None,
        deprecated="Top-level 'env' is deprecated. Please use 'build.env' instead.",
    )

    def only_non_defaults(self, data: dict) -> dict:
        """Return a dictionary containing only non-default values from the recipe.

        Parameters
        ----------
        data : dict
            The recipe data as a dictionary.

        Returns
        -------
        dict
            A dictionary containing only non-default values.
        """

        defaults = Recipe(dates={"values": []}).model_dump()
        output_defaults = {
            "gridded": GriddedOutput().model_dump(),
            "tabular": TabularOutput().model_dump(),
            "trajectories": TrajectoriesOutput().model_dump(),
        }

        def _dates_variant(config: dict) -> str:
            if config.get("hindcasts", False):
                return "hindcasts"
            if "values" in config:
                return "values"
            return "start_end"

        def _output_variant(config: dict) -> str:
            return config.get("layout", config.get("format", "gridded"))

        def _only_non_defaults(d, default_d, path: tuple[str, ...] = ()):

            if type(d) is not type(default_d):
                return d

            if isinstance(d, dict):
                # Output is a discriminated union. Compare against defaults of
                # the active variant so we keep its discriminator naturally,
                # without dropping then re-injecting it later.
                if path == ("output",):
                    variant = _output_variant(d)
                    if variant not in output_defaults:
                        return d
                    default_d = output_defaults[variant]

                # Dates is a discriminated union. If the recipe uses a
                # different variant than the synthetic default (values), keep
                # the whole section so variant-specific keys are preserved.
                if path == ("dates",) and _dates_variant(d) != _dates_variant(default_d):
                    return d

                res = d.copy()
                for k, v in list(d.items()):
                    if k not in default_d:
                        del res[k]
                        continue

                    if v == default_d[k]:
                        del res[k]
                        continue

                    res[k] = _only_non_defaults(v, default_d[k], path + (k,))
                return res

            return d

        return _only_non_defaults(data, defaults)

    def strip_unknown_keys(self, data: dict) -> dict:
        assert isinstance(data, dict)
        defaults = Recipe(input={"empty": {}}, dates={"values": []}).model_dump()
        result = {key: data[key] for key in defaults.keys() if key in data}
        # Trajectory-only keys are omitted when unused, so gridded/tabular
        # recipes keep the same metadata shape they had before these fields
        # existed.
        for key in ("base_dates", "steps"):
            if result.get(key) is None:
                result.pop(key, None)
        return result

    def make_groups(self) -> "Groups | TrajectoryGroups":
        """Build the appropriate Groups object for this recipe.

        Returns
        -------
        Groups or TrajectoryGroups
            Date groups matching the recipe output layout.
        """
        if isinstance(self.output, TrajectoriesOutput):
            from anemoi.datasets.dates.groups import TrajectoryGroups

            return TrajectoryGroups(
                steps=self.steps,
                group_by=self.build.group_by,
                base_dates=self.base_dates,
            )

        from anemoi.datasets.dates.groups import Groups

        return Groups(self.dates, group_by=self.build.group_by)


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
