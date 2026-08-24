# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import glob
import logging
import os
from functools import cached_property
from typing import Any

import numpy as np
import tqdm
from anemoi.transform.variables import Variable
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.buffering import ReadAheadBuffer
from anemoi.datasets.create.recipe.dates import TrajectoryDates
from anemoi.datasets.dates.groups import TrajectoryGroups

from ..dataset import Dataset
from ..gridded.creator import GriddedCreator
from ..statistics import TrajectoryStatisticsCollector
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
        return self.recipe.make_groups()

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
            frequency=provider.frequency,  # base-date frequency
            step_frequency=provider.steps.frequency,  # forecast-step frequency
            layout="trajectories",
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
        metadata["steps"] = self.recipe.steps.model_dump(mode="json")

        metadata["dimensions"] = ["base_dates", "variables", "ensembles", "steps", "values"]

        # Base dates are trajectory-specific metadata.
        base_dates = self._metadata_dates()
        metadata["start_base_date"] = base_dates[0].isoformat()
        metadata["end_base_date"] = base_dates[-1].isoformat()

        # The statistics envelope is defined on valid times (base date + step),
        # matching ``TrajectoryStatisticsFilter`` and the behaviour of
        # ``open_dataset(start=, end=)``: a trajectory is kept only when its
        # whole envelope lies within the statistics interval.  Overwrite the
        # base-date-derived values written by ``super().collect_metadata``.
        provider = self.groups.provider
        assert isinstance(provider, TrajectoryDates), type(provider)
        _, steps = provider.factorise()
        stats_filter = self.recipe.statistics.trajectory_statistics_filter(
            np.array(base_dates, dtype="datetime64[s]"), steps
        )
        metadata["statistics_start_date"] = stats_filter.statistics_start_date
        metadata["statistics_end_date"] = stats_filter.statistics_end_date

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

        # ``load_result`` streams the write one ``(base_date, step)`` region at a
        # time so that only a single forecast step -- not the whole trajectory --
        # is held in memory. That is only chunk-aligned when the base-date and
        # step axes are chunked by 1 (see ``_check_streaming_chunks``).
        self._check_streaming_chunks(list(coords), chunks)

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

    @staticmethod
    def _check_streaming_chunks(dims: list[str], chunks: tuple[int, ...]) -> None:
        """Assert the base-date and step axes are chunked by 1.

        The write in :meth:`load_result` is streamed one ``(base_date, step)``
        region at a time so that only a single forecast step -- not the whole
        trajectory -- is held in memory.  Each region maps onto whole chunks
        (written exactly once, no read-modify-write of an already-populated
        chunk) only when both the base-date and step axes are chunked by 1.
        Both default to 1; forbid any other value rather than silently degrade.

        Parameters
        ----------
        dims : list of str
            The dimension names in array order.
        chunks : tuple of int
            The chunk sizes in array order.
        """
        for axis_name, config_key in (("base_dates", "base_dates"), ("steps", "steps")):
            chunk = chunks[dims.index(axis_name)]
            if chunk != 1:
                raise ValueError(
                    f"The 'trajectories' layout requires 'output.chunking.{config_key}' == 1, got {chunk}."
                )

    def load_result(self, result: Any, dataset: Dataset) -> None:
        """Load a multi-basetime, multi-step forecast cube into the 5-D dataset array.

        The whole trajectory is retrieved with a single request, but the write
        is *streamed* one ``(base_date, step)`` region at a time so that only a
        single forecast step is held in memory rather than the whole trajectory.

        The cube's outermost axis is ``traj_point`` (``{date}_{time}_{step}``,
        see ``TrajectoryGriddedContext.order_by``), so all cubelets sharing a
        ``(basetime, step)`` are contiguous.  We accumulate their fields into a
        single ``(variables, ensembles, cells)`` step_block -- keyed by the
        ``(basetime, step, variable, ensemble)`` identifiers read from the field
        metadata -- and flush it to ``data[date_idx, :, :, step_idx, :]`` as
        soon as the region changes.  Because the base-date and step axes are
        chunked by 1 (enforced in :meth:`initialise_dataset`), each region maps
        onto whole chunks and is written exactly once, with no read-modify-write.

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
        all_steps = dataset.steps  # numpy timedelta64  – authoritative

        assert np.array_equal(
            np.array(base_dates_from_provider, dtype="datetime64[s]"), all_basetimes
        ), "provider.factorise() base_dates do not match dataset.base_dates"
        assert np.array_equal(steps_from_provider, all_steps), "provider.factorise() steps do not match dataset.steps"

        # Coverage guard (mirrors the gridded creator's ``check_shape``): the
        # cube's leading axis is ``traj_point`` -- one entry per unique
        # ``(basetime, step)`` pair actually present in the retrieved fields. If
        # a whole basetime/step is missing across all variables (e.g. a MARS
        # gap), the cube is smaller than the group and those slots would stay
        # NaN. Fail loudly instead.
        expected_traj_points = len(list(result.group_of_dates))
        got_traj_points = cube.extended_user_shape[0]
        if got_traj_points != expected_traj_points:
            raise ValueError(
                f"Trajectories: cube has {got_traj_points} (basetime, step) points, "
                f"expected {expected_traj_points} for this group -- some fields are missing."
            )

        variables = list(dataset.get_metadata("variables"))
        var_to_idx = {v: i for i, v in enumerate(variables)}

        # Use the same remapping (``build.variable_naming``) that named the
        # variables at init time, so e.g. wave spectra are recovered as
        # ``{param}_{directionNumber}_{frequencyNumber}`` rather than bare
        # ``param``/``param_levelist``.
        remapping = result.context.remapping

        # Map GRIB member numbers to positions on the ensemble axis.  Member
        # numbering schemes vary (0-based with control, 1-based perturbed
        # members, ...), so the number cannot be used as an index directly.
        numbers = cube.user_coords.get("number")
        number_to_index = None if numbers is None else {int(n): i for i, n in enumerate(numbers)}

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

        # Stream the write one region at a time.  Only one ``(base_date, step)``
        # step_block is resident at a time; ``traj_point`` being the outermost
        # cube axis guarantees each region is visited exactly once and
        # contiguously.
        data_array = dataset.data
        assert (
            data_array.chunks[0] == 1 and data_array.chunks[3] == 1
        ), f"Trajectory streaming write requires base_dates/steps chunks == 1, got {data_array.chunks}"
        n_variables = len(variables)
        n_ensembles = data_array.shape[2]
        n_cells = data_array.shape[4]

        current_key: tuple[int, int] | None = None
        step_block: Any = None
        flushed_keys: set[tuple[int, int]] = set()

        def flush(key: tuple[int, int] | None, block: Any) -> None:
            if key is not None:
                date_idx, step_idx = key
                data_array[date_idx, :, :, step_idx, :] = block

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

            # Recover variable name via the build.variable_naming remapping
            # (param_level pattern), matching the names declared at init.
            var_name = remapping(field.metadata)("param_level", default=None)
            if var_name is None:
                var_name = str(field.metadata("param"))

            number = field.metadata("number", default=0) or 0
            if number_to_index is None:
                ens_idx = 0
            else:
                ens_idx = number_to_index.get(int(number))
                if ens_idx is None:
                    raise ValueError(
                        f"Trajectories: member number {number} not in the cube's "
                        f"'number' coordinate {list(numbers)}"
                    )

            date_idx = basetime_index_of(basetime)
            step_idx = step_index_of(step_td)

            var_idx = var_to_idx.get(var_name)
            if var_idx is None:
                raise ValueError(f"Trajectories: field variable {var_name!r} not in dataset variables")

            key = (date_idx, step_idx)
            if key != current_key:
                # Region changed: flush the previous step_block and start a new one.
                flush(current_key, step_block)
                if current_key is not None:
                    flushed_keys.add(current_key)
                assert key not in flushed_keys, (
                    f"Trajectories: (base_date, step) region {key} revisited; the cube is "
                    "not ordered by 'traj_point' as expected, so streaming writes would "
                    "overwrite an already-flushed region."
                )
                current_key = key
                step_block = np.full((n_variables, n_ensembles, n_cells), np.nan, dtype=data_array.dtype)

            step_block[var_idx, ens_idx, :] = data

        # Flush the final region.
        flush(current_key, step_block)

        # Detect silently-missing data. A source that returns no fields for this
        # group (e.g. a corrupt earthkit cache, a MARS gap, or an over-narrow
        # filter) drops its variables from the cube, leaving NaN holes in the
        # dataset. The gridded creator guards against this via
        # ``Variable.check_compatibility``; mirror it here so trajectories fail
        # loudly instead of writing a corrupt dataset. ``result.typed_variables``
        # is derived from the actually-retrieved fields, ``dataset.typed_variables``
        # from the variables declared at init.
        Variable.check_compatibility(dataset.typed_variables, result.typed_variables)

    def context(self):
        return TrajectoryGriddedContext(self.recipe)

    def group_shapes(self) -> list[tuple[int, ...]]:
        """Return the shape of each group's footprint in the data array.

        Each entry is ``(n_base_dates, n_steps)``.  The first element is
        used by ``group_to_range`` to index axis 0.
        """
        shapes = []
        for g in self.groups:
            n_base_dates = len(set(bt for _, bt in g))
            n_steps = len(set(vt - bt for vt, bt in g)) if len(g) > 0 else 0
            shapes.append((n_base_dates, n_steps))
        return shapes

    def _tendencies_to_compute(self, dataset: Dataset) -> dict[str, int]:
        """Return ``{name: step_delta}`` for the tendencies to compute.

        The reference frequency is the **step spacing** of the trajectory,
        not the base-date frequency, because tendencies are computed along
        the step axis within a single trajectory.  Deltas that are not whole
        multiples of the step spacing are skipped with a warning, mirroring
        the gridded behaviour.
        """
        tendencies_config = self.recipe.statistics.tendencies
        if tendencies_config is None:
            # Tendencies are disabled by default.
            tendencies_config = False
        if tendencies_config is True:
            tendencies_list = [1, 3, 6, 12, 24]
        elif tendencies_config is False:
            return {}
        else:
            tendencies_list = list(tendencies_config)

        steps = dataset.steps
        if len(steps) < 2:
            LOG.warning("Trajectory has fewer than 2 steps; cannot compute tendencies.")
            return {}

        step_diffs = np.diff(steps)
        if not np.all(step_diffs == step_diffs[0]):
            LOG.warning("Trajectory steps are not uniformly spaced; skipping tendency statistics.")
            return {}

        step_spacing_seconds = int(step_diffs[0].astype("timedelta64[s]").astype("int64"))
        step_spacing_td = datetime.timedelta(seconds=step_spacing_seconds)

        tendencies: dict[str, int] = {}
        for delta in tendencies_list:
            td = frequency_to_timedelta(delta)
            ratio = td / step_spacing_td
            if int(ratio) == ratio:
                tendencies[frequency_to_string(td)] = int(ratio)
            else:
                LOG.warning(
                    f"Tendency delta {delta} is not a multiple of trajectory step spacing "
                    f"{frequency_to_string(step_spacing_td)}, skipping."
                )

        return tendencies

    def _compute_partial_statistics(self, dataset: Dataset, start, end) -> TrajectoryStatisticsCollector:
        base_dates = dataset.base_dates
        steps = dataset.steps

        collector = TrajectoryStatisticsCollector(
            variables_names=self.variables_names,
            filter=self.recipe.statistics.trajectory_statistics_filter(base_dates, steps),
            tendencies=self._tendencies_to_compute(dataset),
        )

        data = ReadAheadBuffer(dataset.data, start=start)
        chunk_size = data.chunks[0]

        for i in tqdm.tqdm(range(start, end, chunk_size)):
            j = min(i + chunk_size, end)
            chunk = data[i:j]
            collector.collect(chunk, base_dates[i:j])

        return collector

    def compute_and_store_statistics(self, dataset: Dataset) -> None:
        if os.path.exists(self.work_dir):
            precomputed = list(glob.glob(os.path.join(self.work_dir, "statistics_*.pkl")))
            if precomputed:
                LOG.info(f"Loading precomputed statistics from {self.work_dir} ({len(precomputed):,} files)")
                collector = TrajectoryStatisticsCollector.load_precomputed(dataset, precomputed)
                collector.add_to_dataset(dataset)
                return

        LOG.info("Computing statistics for the full trajectory dataset")
        collector = self._compute_partial_statistics(dataset, 0, dataset.data.shape[0])
        collector.add_to_dataset(dataset)
