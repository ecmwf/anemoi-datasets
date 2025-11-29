# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import time
from typing import Any

import numpy as np
import tqdm
from anemoi.utils.dates import as_datetime
from anemoi.utils.humanize import compress_dates
from anemoi.utils.humanize import seconds_to_human

from anemoi.datasets.create.check import check_data_values
from anemoi.datasets.create.chunks import ChunkFilter
from anemoi.datasets.create.statistics import compute_statistics
from anemoi.datasets.create.writer import ViewCacheArray

from ..base.load import LoadTask
from .tasks import GriddedTaskMixin
from .tasks import WritableDataset

LOG = logging.getLogger(__name__)


class Load(LoadTask, GriddedTaskMixin):
    """A class to load data into a dataset."""

    def __init__(
        self,
        path: str,
        config: dict | None = None,
        parts: str | None = None,
        use_threads: bool = False,
        statistics_temp_dir: str | None = None,
        progress: Any = None,
        cache: str | None = None,
        **kwargs: Any,
    ):
        """Initialize a Load instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        parts : Optional[str], optional
            The parts to load.
        use_threads : bool, optional
            Whether to use threads.
        statistics_temp_dir : Optional[str], optional
            The directory for temporary statistics.
        progress : Any, optional
            The progress indicator.
        cache : Optional[str], optional
            The cache directory.
        """
        super().__init__(path, config=config, cache=cache)
        self.use_threads = use_threads
        self.statistics_temp_dir = statistics_temp_dir
        self.progress = progress
        self.parts = parts
        self.dataset = WritableDataset(self.path)

        self.main_config = self.dataset.get_main_config()
        self.create_elements(self.main_config)
        self.read_dataset_metadata(self.dataset.path)

        total = len(self.registry.get_flags())
        self.chunk_filter = ChunkFilter(parts=self.parts, total=total)

        self.data_array = self.dataset.data_array
        self.n_groups = len(self.groups)

    def _run(self) -> None:
        """Internal method to run the data loading."""
        for igroup, group in enumerate(self.groups):
            if not self.chunk_filter(igroup):
                continue
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={len(self.groups)} (already done)")
                continue

            # assert isinstance(group[0], datetime.datetime), type(group[0])
            LOG.debug(f"Building data for group {igroup}/{self.n_groups}")

            result = self.input.select(self.context, argument=group)
            assert result.group_of_dates == group, (len(result.group_of_dates), len(group), group)

            # There are several groups.
            # There is one result to load for each group.
            self.load_result(result)
            self.registry.set_flag(igroup)

        self.registry.add_provenance(name="provenance_load")
        self.tmp_statistics.add_provenance(name="provenance_load", config=self.main_config)

        self.dataset.print_info()

    def load_result(self, result: Any) -> None:
        """Load the result into the dataset.

        Parameters
        ----------
        result : Any
            The result to load.
        """
        # There is one cube to load for each result.
        dates = list(result.group_of_dates)

        LOG.debug(f"Loading cube for {len(dates)} dates")

        cube = result.get_cube()
        shape = cube.extended_user_shape
        dates_in_data = cube.user_coords["valid_datetime"]

        LOG.debug(f"Loading {shape=} in {self.data_array.shape=}")

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

        indexes = dates_to_indexes(self.dates, dates_in_data)

        array = ViewCacheArray(self.data_array, shape=shape, indexes=indexes)
        LOG.info(f"Loading array shape={shape}, indexes={len(indexes)}")
        self.load_cube(cube, array)

        stats = compute_statistics(array.cache, self.variables_names, allow_nans=self._get_allow_nans())
        self.tmp_statistics.write(indexes, stats, dates=dates_in_data)
        LOG.info("Flush data array")
        array.flush()
        LOG.info("Flushed data array")

    def _get_allow_nans(self) -> bool | list:
        """Get the allow_nans configuration.

        Returns
        -------
        bool | list
            The allow_nans configuration.
        """
        config = self.main_config
        if "allow_nans" in config.build:
            return config.build.allow_nans

        return config.statistics.get("allow_nans", [])

    def load_cube(self, cube: Any, array: ViewCacheArray) -> None:
        """Load the cube into the array.

        Parameters
        ----------
        cube : Any
            The cube to load.
        array : ViewCacheArray
            The array to load into.
        """
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

            name = self.variables_names[local_indexes[1]]
            check_data_values(
                data[:],
                name=name,
                log=[i, data.shape, local_indexes],
                allow_nans=self._get_allow_nans(),
            )

            now = time.time()
            array[local_indexes] = data
            save += time.time() - now

        now = time.time()
        save += time.time() - now
        LOG.debug(
            f"Elapsed: {seconds_to_human(time.time() - start)}, "
            f"load time: {seconds_to_human(load)}, "
            f"write time: {seconds_to_human(save)}."
        )


task = Load
