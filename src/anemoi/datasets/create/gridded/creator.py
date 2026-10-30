# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import glob
import logging
import os
import time
from abc import abstractmethod
from typing import Any

import numpy as np
import tqdm
from anemoi.transform.variables import Variable
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.humanize import compress_dates
from anemoi.utils.humanize import seconds_to_human

from anemoi.datasets.buffering import ReadAheadBuffer
from anemoi.datasets.buffering import WriteBehindBuffer
from anemoi.datasets.compat import blosc_compressor
from anemoi.datasets.compat import get_zarr_format

from ..creator import Creator
from ..dataset import Dataset
from ..parts import part_bounds
from ..statistics import StatisticsCollector
from .context import SimpleGriddedContext

LOG = logging.getLogger(__name__)


class GriddedCreator(Creator):

    def _metadata_dates(self):
        """Return the datetime values used for statistics metadata.

        Trajectories override this to return the base dates (factorised from
        ``(basetime, step)`` tuples).
        """
        return self.groups.provider.full_dates_list()

    def _metadata_date_range(self):
        """Return ``(start_date, end_date)`` for the dataset metadata.

        The default implementation uses the first/last of
        :meth:`_metadata_dates`.  Trajectories override this to return the
        envelope (basetime + step).
        """
        dates = self._metadata_dates()
        return dates[0], dates[-1]

    def collect_metadata(self, metadata: dict):
        """Run the initialisation process for the dataset."""
        super().collect_metadata(metadata)

        dates = self._metadata_dates()
        frequency = self.groups.provider.frequency
        missing = self.groups.provider.missing

        variables = self.minimal_input.variables
        LOG.info(f"Found {len(variables)} variables : {', '.join(variables)}.")

        metadata["remapping"] = self.recipe.build.remapping
        # Read from the context rather than the recipe so that layouts whose
        # cube ordering is internal (e.g. trajectories) can provide their own
        # ``order_by`` without exposing it in the recipe schema.
        metadata["order_by"] = self.context().order_by

        metadata["dimensions"] = ["dates", "variables", "ensembles", "values"]

        metadata["resolution"] = self.minimal_input.resolution

        metadata["data_request"] = self.minimal_input.data_request
        metadata["field_shape"] = self.minimal_input.field_shape
        metadata["proj_string"] = self.minimal_input.proj_string

        metadata["variables_metadata"] = self.minimal_input.variables_metadata

        # TODO: below may be common with tabular
        metadata["variables"] = variables

        start_date, end_date = self._metadata_date_range()
        metadata["start_date"] = start_date.isoformat()
        metadata["end_date"] = end_date.isoformat()
        metadata["frequency"] = frequency
        metadata["missing_dates"] = [_.isoformat() for _ in missing]

        # TODO:
        # ``dates`` are the values along the dataset's time axis: valid dates
        # for gridded, base dates for trajectories.  ``statistics_dates``
        # must receive the same kind so that the resulting filter boundaries
        # are consistent with ``dataset.dates`` used in
        # ``_compute_partial_statistics``.
        # This is for simplicity, for now, but they should arguably be valid dates
        # in both cases, statistics should be computed with data selection based
        # on valid dates, for instance setting the default statistics period to
        # the 20 most recent years of valid data should remove any data
        # that is not valid in this date range.
        # but this is not implemented yet.
        # related to the start and end being the "envelope" of the dates in the dataset.
        start, end = self.recipe.statistics.statistics_dates(dates)

        metadata["statistics_start_date"] = start
        metadata["statistics_end_date"] = end

        # metadata["variables_with_nans"] = variables_with_nans
        metadata["allow_nans"] = self.recipe.build.allow_nans

    @abstractmethod
    def initialise_dataset(self, dataset) -> None:
        pass

    def load_result(self, result: Any, dataset: Dataset) -> None:
        """Load the result into the dataset."""
        # There is one cube to load for each result.
        dates = list(result.group_of_dates)

        LOG.info(f"Loading cube for {len(dates)} dates")

        cube = result.get_cube()
        shape = cube.extended_user_shape
        dates_in_data = cube.user_coords["valid_datetime"]

        # LOG.debug(f"Loading {shape=} in {self.data_array.shape=}")

        def check_shape(cube, dates, dates_in_data):
            if cube.extended_user_shape[0] != len(dates):
                print(
                    f"Cube shape does not match the number of dates got {cube.extended_user_shape[0]}, expected {len(dates)}"
                )
                print("Requested dates", compress_dates(dates))
                print("Cube dates", compress_dates(dates_in_data))

                a = {as_datetime(_) for _ in dates}
                b = {as_datetime(_) for _ in dates_in_data}

                print("Missing dates", compress_dates(a - b))
                print("Extra dates", compress_dates(b - a))

                raise ValueError(
                    f"Cube shape does not match the number of dates got {cube.extended_user_shape[0]}, expected {len(dates)}"
                )

        check_shape(cube, dates, dates_in_data)

        def check_dates_in_data(dates_in_data, requested_dates):
            _requested_dates = [np.datetime64(_) for _ in requested_dates]
            _dates_in_data = [np.datetime64(_) for _ in dates_in_data]
            if _dates_in_data != _requested_dates:
                LOG.error("Dates in data are not the requested ones:")

                dates_in_data = set(dates_in_data)
                requested_dates = set(requested_dates)

                missing = sorted(requested_dates - dates_in_data)
                extra = sorted(dates_in_data - requested_dates)

                if missing:
                    LOG.error(f"Missing dates: {[_.isoformat() for _ in missing]}")
                if extra:
                    LOG.error(f"Extra dates: {[_.isoformat() for _ in extra]}")

                raise ValueError("Dates in data are not the requested ones")

        check_dates_in_data(dates_in_data, dates)

        def dates_to_indexes(dates, all_dates):
            x = np.array(dates, dtype=np.datetime64)
            y = np.array(all_dates, dtype=np.datetime64)
            bitmap = np.isin(x, y)
            return np.where(bitmap)[0]

        indexes = dates_to_indexes(dataset.dates, dates_in_data)

        with WriteBehindBuffer(dataset.data) as array:
            LOG.info(f"Loading array shape={shape}, indexes={len(indexes)}")
            self._load_cube(cube, array, indexes)

        # The units have been set during the first load, so we can check for consistency here
        old = dataset.typed_variables
        new = result.typed_variables

        Variable.check_compatibility(old, new)

    def check_dataset_name(self, path: str) -> None:
        from pathlib import Path

        from ..naming import check_dataset_name

        name = Path(path).stem

        dates = self.groups.provider.values
        frequency = self.groups.provider.frequency

        for message in check_dataset_name(
            name,
            resolution=self.minimal_input.resolution,
            start_date=dates[0],
            end_date=dates[-1],
            frequency=frequency,
            layout="gridded",
        ):
            LOG.warning("Dataset name warning: %s", message)

    ######################################################

    @abstractmethod
    def context(self):
        pass

    def _load_cube(self, cube: Any, array: Any, indexes: Any) -> None:
        """Load the cube into the array."""
        # There are several cubelets for each cube
        start = time.time()
        load = 0
        save = 0

        reading_chunks = None
        total = cube.count(reading_chunks)
        LOG.debug(f"Loading datacube: {cube}")

        def position(x: Any) -> int | None:
            if isinstance(x, str) and "/" in x:
                x = x.split("/")
                return int(x[0])
            return None

        bar = tqdm.tqdm(
            iterable=cube.iterate_cubelets(reading_chunks),
            total=total,
            desc=f"Loading datacube {cube}",
            position=position(self.parts),
        )
        for i, cubelet in enumerate(bar):
            bar.set_description(f"Loading {i}/{total}")

            now = time.time()
            data = cubelet.to_numpy()
            local_indexes = cubelet.coords
            load += time.time() - now

            global_index = (int(indexes[local_indexes[0]]),) + local_indexes[1:]

            now = time.time()
            array[global_index] = data
            save += time.time() - now

        now = time.time()
        save += time.time() - now
        LOG.debug(
            f"Elapsed: {seconds_to_human(time.time() - start)}, "
            f"load time: {seconds_to_human(load)}, "
            f"write time: {seconds_to_human(save)}."
        )

    def finalise_dataset(self, dataset: Dataset) -> None:
        # Nothing to do here
        pass

    def _tendencies_to_compute(self, dataset: Dataset) -> dict[str, int]:
        """Return the tendencies to compute for the dataset, based on the recipe configuration."""
        frequency = dataset.frequency

        tendencies_config = self.recipe.statistics.tendencies
        if tendencies_config is None:
            # Tendencies are disabled by default.
            tendencies_config = False
        if tendencies_config is True:
            tendencies_list = [1, 3, 6, 12, 24]
        elif tendencies_config is False:
            tendencies_list = []
        else:
            tendencies_list = list(tendencies_config)

        tendencies = {}
        for delta in tendencies_list:
            td = frequency_to_timedelta(delta)
            ratio = td / frequency
            if int(ratio) == ratio:
                tendencies[frequency_to_string(td)] = int(ratio)
            else:
                LOG.warning(
                    f"Tendency delta {delta} is not a multiple of dataset frequency "
                    f"{frequency_to_string(frequency)}, skipping."
                )

        return tendencies

    def _compute_partial_statistics(self, dataset: Dataset, lo: int, hi: int) -> StatisticsCollector:
        """Compute the statistics of the rows ``[lo, hi)`` of the dataset.

        The ``D`` rows preceding ``lo`` (``D`` being the largest tendency delta, in rows) are
        read first and only used to prime the tendency windows, so that the tendencies of the
        first rows of ``[lo, hi)`` are computed. The collectors of contiguous ranges covering
        the dataset can therefore be merged without any further adjustment.

        Parameters
        ----------
        dataset : Dataset
            The dataset to read.
        lo : int
            First row of the range.
        hi : int
            End of the range (exclusive).

        Returns
        -------
        StatisticsCollector
            The collector for the range.
        """
        dates = dataset.dates
        tendencies = self._tendencies_to_compute(dataset)

        collector = StatisticsCollector(
            variables_names=self.variables_names,
            filter=self.recipe.statistics.statistics_filter(dates),
            tendencies=tendencies,
        )

        if hi <= lo:
            return collector

        # Warm-up rows, only needed for tendencies. On-disk dates include the
        # missing dates (as NaN rows), so a delta in rows is a delta in time.
        warm_up = max(lo - max(tendencies.values(), default=0), 0)

        # Batches are aligned on the chunks of the data array, and split at `lo`.
        step = dataset.data.chunks[0]
        bounds = sorted({warm_up, lo, hi} | set(range((warm_up // step + 1) * step, hi, step)))

        data = ReadAheadBuffer(dataset.data, start=warm_up // step)

        for i, j in tqdm.tqdm(list(zip(bounds[:-1], bounds[1:])), desc=f"Statistics [{lo}:{hi})"):
            if j <= lo:
                collector.prime(data[i:j], dates[i:j])
            else:
                collector.collect(data[i:j], dates[i:j])

        return collector

    def _statistics_part_path(self, index: int, lo: int, hi: int) -> str:
        return os.path.join(self.work_dir, f"statistics_part_{index:06d}-{lo:09d}-{hi:09d}.pkl")

    def _statistics_part_files(self) -> list[str]:
        # Loader pickles written by older versions (statistics_{group}-...pkl) are ignored.
        return sorted(glob.glob(os.path.join(self.work_dir, "statistics_part_*.pkl")))

    def finalise_prepare(self, dataset: Dataset) -> None:
        """Prepare stage: check that the dataset is fully loaded and remove stale partial statistics.

        Parameters
        ----------
        dataset : Dataset
            The dataset being finalised.
        """
        self.finalise_dataset(dataset)

        if "_build" in dataset.store and not dataset.ready():
            raise RuntimeError(
                f"Dataset {self.path} is not fully loaded ({dataset.todo_remaining()} groups remaining), "
                "run `anemoi-datasets load` before `finalise`."
            )

        # A previous run may have used a different number of parts.
        stale = self._statistics_part_files() + glob.glob(os.path.join(self.work_dir, "statistics_part_*.pkl.tmp"))
        for path in stale:
            os.unlink(path)
        if stale:
            LOG.info(f"Removed {len(stale)} stale partial statistics file(s) from {self.work_dir}")

    def finalise_load(self, dataset: Dataset) -> None:
        """Load stage: compute the statistics of the rows of this process's ``--parts``.

        The rows are split in contiguous ranges of whole chunks. The partial statistics are
        written to the work directory, and merged by the tidy stage.

        Parameters
        ----------
        dataset : Dataset
            The dataset being finalised.
        """
        nrows = dataset.data.shape[0]
        chunk_rows = dataset.data.chunks[0]
        nchunks = (nrows + chunk_rows - 1) // chunk_rows

        index, chunk_lo, chunk_hi = part_bounds(self.parts, nchunks)
        lo = min(chunk_lo * chunk_rows, nrows)
        hi = min(chunk_hi * chunk_rows, nrows)

        path = self._statistics_part_path(index, lo, hi)
        if os.path.exists(path):
            LOG.info(f"Statistics for part {index} (rows [{lo}:{hi})) already computed: {path}")
            return

        LOG.info(f"Computing statistics for part {index} ({self.parts}): rows [{lo}:{hi}) of {nrows}")

        collector = self._compute_partial_statistics(dataset, lo, hi)

        os.makedirs(self.work_dir, exist_ok=True)
        collector.serialise(path + ".tmp", group=index, start=lo, end=hi)
        os.replace(path + ".tmp", path)

    def compute_and_store_statistics(self, dataset: Dataset) -> None:
        precomputed = self._statistics_part_files()
        if precomputed:
            LOG.info(f"Loading precomputed statistics from {self.work_dir} ({len(precomputed):,} files)")
            collector = self._statistics_collector_class().load_precomputed(dataset, precomputed)
            collector.add_to_dataset(dataset)
            return

        if "_build" in dataset.store and dataset.total_todo() > 1:
            # The loaders do not compute statistics: if they ran in parallel, this
            # single process now reads the whole dataset.
            LOG.warning(
                f"Computing the statistics of the full dataset ({dataset.data.shape[0]:,} dates) in a single "
                "process. If the dataset was loaded in parallel (`load --parts`), also run the finalise in "
                "parallel: `finalise --prepare`, then `finalise --load --parts i/n` for each part, then "
                "`finalise --tidy`."
            )

        LOG.info("Computing statistics for the full dataset")
        collector = self._compute_partial_statistics(dataset, 0, dataset.data.shape[0])
        collector.add_to_dataset(dataset)

    def _statistics_collector_class(self) -> type:
        return StatisticsCollector


class SimpleGriddedCreator(GriddedCreator):
    def context(self):
        return SimpleGriddedContext(self.recipe)

    def initialise_dataset(self, dataset: Dataset) -> None:
        all_dates = self.groups.provider.full_dates_list()
        shape = (len(all_dates),) + self.minimal_input.shape[1:]

        assert len(shape) == 4, f"Expected 4D shape, got {shape}"

        coords = self.minimal_input.coords
        coords["dates"] = all_dates
        chunks = self.recipe.output.get_chunking(coords)

        grid_points = self.minimal_input.grid_points

        # Create arrays

        dataset.add_array(
            name="data",
            chunks=chunks,
            dtype=self.recipe.output.dtype,
            shape=shape,
            dimensions=("time", "variable", "ensemble", "cell"),
            fill_value=np.nan,
            # Explicit, so that every Zarr format writes the same codec: without it, format 2 arrays get Blosc(lz4, 5, byte shuffle)
            # and format 3 arrays get zstd without shuffle.
            compressors=blosc_compressor(cname="lz4", clevel=5, shuffle=1, zarr_format=get_zarr_format(dataset.store)),
        )

        dataset.add_array(name="dates", data=np.array(all_dates, "<M8[s]"), dimensions=("time",))
        dataset.add_array(name="latitudes", data=grid_points[0], dimensions=("cell",))
        dataset.add_array(name="longitudes", data=grid_points[1], dimensions=("cell",))
