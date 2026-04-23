# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from typing import Any

import numpy as np

from anemoi.datasets.buffering import WriteBehindBuffer
from anemoi.datasets.dates import TrajectoryDates
from anemoi.datasets.dates.groups import TrajectoryGroups

from ..dataset import Dataset
from ..gridded.creator import GriddedCreator
from .context import TrajectoryGriddedContext

LOG = logging.getLogger(__name__)


class TrajectoryGriddedCreator(GriddedCreator):

    @cached_property
    def groups(self) -> TrajectoryGroups:
        """Return the date groups for the trajectories dataset.

        Overrides the base implementation so that ``self.groups.provider`` is a
        :class:`TrajectoryDates` whose ``values`` are ``(basetime, step)``
        pairs; basetimes and steps are recovered via
        :meth:`TrajectoryDates.factorise`.
        """
        return TrajectoryGroups(
            steps=self.recipe.steps,
            group_by=self.recipe.build.group_by,
            base_dates=self.recipe.base_dates,
        )

    def check_dataset_name(self, path: str) -> None:
        from pathlib import Path

        from ..naming import check_dataset_name

        provider = self.groups.provider
        assert isinstance(provider, TrajectoryDates), type(provider)

        name = Path(path).stem
        base_dates, _ = provider.factorise()

        for message in check_dataset_name(
            name,
            resolution=self.minimal_input.resolution,
            start_date=base_dates[0],
            end_date=base_dates[-1],
            frequency=provider.frequency,
        ):
            LOG.warning("Dataset name warning: %s", message)

    def _metadata_date_range(self):
        """Return the envelope ``(start_date, end_date)`` for trajectories.

        ``start_date`` is the earliest valid time (first basetime + first step),
        ``end_date`` is the latest valid time (last basetime + last step).
        """
        import datetime

        provider = self.groups.provider
        assert isinstance(provider, TrajectoryDates), type(provider)
        base_dates, steps = provider.factorise()

        step_start = datetime.timedelta(seconds=int(steps[0] / np.timedelta64(1, "s")))
        step_end = datetime.timedelta(seconds=int(steps[-1] / np.timedelta64(1, "s")))

        return base_dates[0] + step_start, base_dates[-1] + step_end

    def collect_metadata(self, metadata: dict):
        """Collect metadata for the trajectories dataset."""
        super().collect_metadata(metadata)
        metadata["layout"] = "trajectories"
        metadata["steps"] = self.recipe.steps
        # Variables live on axis 1 (same as gridded); ensembles on axis 2; steps on axis -2.
        metadata["ensemble_dimension"] = 2
        metadata["step_dimension"] = -2

        # Base dates are trajectory-specific metadata.
        base_dates = self._metadata_dates()
        metadata["start_base_date"] = base_dates[0].isoformat()
        metadata["end_base_date"] = base_dates[-1].isoformat()

    def _metadata_dates(self):
        """Return the sorted-unique base dates (not ``(basetime, step)`` tuples)."""
        provider = self.groups.provider
        assert isinstance(provider, TrajectoryDates), type(provider)
        base_dates, _ = provider.factorise()
        return base_dates

    def initialise_dataset(self, dataset: Dataset) -> None:
        provider = self.groups.provider
        assert isinstance(provider, TrajectoryDates), type(provider)

        base_dates, steps = provider.factorise()

        # On-disk layout: (base_dates, variables, ensembles, steps, cells).
        mi_shape = self.minimal_input.shape
        assert len(mi_shape) == 4, f"Expected 4D minimal_input.shape, got {mi_shape}"
        shape = (len(base_dates),) + mi_shape[1:-1] + (len(steps),) + mi_shape[-1:]

        # Coords dict must be in the same order as the array dimensions so that
        # ``get_chunking`` maps chunk sizes correctly.
        mi_coords = self.minimal_input.coords
        coords = {
            "base_dates": base_dates,
            "variables": mi_coords["variables"],
            "ensembles": mi_coords["ensembles"],
            "steps": mi_coords["steps"],
            "values": mi_coords["values"],
        }
        chunks = self.recipe.output.get_chunking(coords)

        grid_points = self.minimal_input.grid_points

        # Create arrays

        dataset.add_array(
            name="data",
            chunks=chunks,
            dtype=self.recipe.output.dtype,
            shape=shape,
            dimensions=("time", "variable", "ensemble", "step", "cell"),
            fill_value=np.nan,
        )

        dataset.add_array(name="base_dates", data=np.array(base_dates, "<M8[s]"), dimensions=("time",))
        dataset.add_array(name="steps", data=np.array(steps), dimensions=("step",))
        dataset.add_array(name="latitudes", data=grid_points[0], dimensions=("cell",))
        dataset.add_array(name="longitudes", data=grid_points[1], dimensions=("cell",))

    def load_result(self, result: Any, dataset: Dataset) -> None:
        """Load a multi-basetime, multi-step forecast cube into the 5-D dataset array.

        The cube is ordered by ``(date, time, step, param_level, number)``
        (see ``TrajectoriesOutput.order_by``).  Some axes may be squeezed
        out of the cube when they have size 1.  For each cubelet we read the
        underlying field metadata to recover the ``(basetime, step, variable,
        ensemble)`` identifiers and write into ``data[date_idx, step_idx,
        var_idx, ens_idx, :]``.

        Parameters
        ----------
        result : TrajectoryGriddedResult
            Result of the input pipeline for the current group.
        dataset : Dataset
            Target zarr dataset.
        """
        import datetime

        cube = result.get_cube()

        provider = self.groups.provider
        assert isinstance(provider, TrajectoryDates), type(provider)
        base_dates_from_provider, steps_from_provider = provider.factorise()

        all_basetimes = dataset.base_dates  # numpy datetime64[s] – authoritative
        all_steps = dataset.steps           # numpy timedelta64  – authoritative

        assert np.array_equal(np.array(base_dates_from_provider, dtype="datetime64[s]"), all_basetimes), (
            "provider.factorise() base_dates do not match dataset.base_dates"
        )
        assert np.array_equal(steps_from_provider, all_steps), (
            "provider.factorise() steps do not match dataset.steps"
        )

        variables = list(dataset.get_metadata("variables"))
        var_to_idx = {v: i for i, v in enumerate(variables)}

        LOG.info(
            "Loading trajectory cube: cube shape=%s, variables=%d, steps=%d",
            cube.extended_user_shape,
            len(variables),
            len(all_steps),
        )

        def basetime_index_of(basetime: datetime.datetime) -> int:
            indices = np.where(all_basetimes == np.datetime64(basetime, "s"))[0]
            if len(indices) == 0:
                raise ValueError(f"Trajectories: basetime {basetime} not in dataset.base_dates")
            return int(indices[0])

        def step_index_of(step_td: datetime.timedelta) -> int:
            step_np = np.timedelta64(int(step_td.total_seconds()), "s")
            indices = np.where(all_steps == step_np)[0]
            if len(indices) == 0:
                raise ValueError(f"Trajectories: step {step_td} not in dataset.steps")
            return int(indices[0])

        with WriteBehindBuffer(dataset.data) as array:
            for cubelet in cube.iterate_cubelets():
                data = cubelet.to_numpy()
                field = cube[cubelet.coords]

                # Recover basetime from field metadata
                date_int = int(field.metadata("date"))  # YYYYMMDD
                time_int = int(field.metadata("time") or 0)  # HHMM
                basetime = datetime.datetime(
                    year=date_int // 10000,
                    month=(date_int // 100) % 100,
                    day=date_int % 100,
                    hour=time_int // 100,
                    minute=time_int % 100,
                )
                step_td = datetime.timedelta(hours=int(field.metadata("step")))

                # Recover variable name via remapping (param_level = param_levelist)
                param = field.metadata("param")
                levelist = field.metadata("levelist", default=None)
                var_name = f"{param}_{levelist}" if levelist is not None else str(param)

                number = field.metadata("number", default=0) or 0
                ens_idx = int(number) if int(number) > 0 else 0

                date_idx = basetime_index_of(basetime)
                step_idx = step_index_of(step_td)

                var_idx = var_to_idx.get(var_name)
                if var_idx is None:
                    raise ValueError(
                        f"Trajectories: field variable {var_name!r} not in dataset variables"
                    )

                array[(date_idx, var_idx, ens_idx, step_idx)] = data

    def context(self):
        return TrajectoryGriddedContext(self.recipe)
