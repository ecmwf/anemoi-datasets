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
import uuid
import warnings
from functools import cached_property
from typing import Any

import numpy as np
import tqdm
import zarr
from anemoi.utils.dates import as_datetime
from anemoi.utils.humanize import compress_dates
from anemoi.utils.humanize import seconds_to_human
from anemoi.utils.sanitise import sanitise
from earthkit.data.core.order import build_remapping

from ..check import check_data_values
from ..creator import Creator
from ..utils import normalize_and_check_dates
from . import Dataset
from . import build_statistics_dates
from .context import GriddedContext
from .writer import ViewCacheArray

LOG = logging.getLogger(__name__)

VERSION = "0.30"


class GriddedCreator(Creator):

    check_name = False

    def task_init(self) -> Dataset:
        """Run the initialisation process for the dataset."""
        dataset = super().task_init()

        dates = self.groups.provider.values
        frequency = self.groups.provider.frequency
        missing = self.groups.provider.missing

        variables = self.minimal_input.variables
        LOG.info(f"Found {len(variables)} variables : {','.join(variables)}.")

        variables_with_nans = self.recipe.statistics.allow_nans

        ensembles = self.minimal_input.ensembles
        LOG.info(f"Found {len(ensembles)} ensembles : {','.join([str(_) for _ in ensembles])}.")

        grid_points = self.minimal_input.grid_points
        LOG.info(f"gridpoints size: {[len(i) for i in grid_points]}")

        resolution = self.minimal_input.resolution
        LOG.info(f"{resolution=}")

        total_shape, chunks = self.shape_and_chunks(dates)
        dtype = self.output.dtype

        LOG.info(f"Creating Dataset '{self.path}', with {total_shape=}, {chunks=} and {dtype=}")

        metadata = {}
        metadata["uuid"] = str(uuid.uuid4())

        # metadata.update(self.recipe.get("add_metadata", {}))

        # We use model_dump_json to have a JSON string, because Zarr sorts attrs keys
        metadata["_recipe"] = self.recipe.model_dump_json()

        recipe = sanitise(self.recipe.model_dump())

        # Remove stuff added by prepml
        for k in [
            "build_dataset",
            "config_format_version",
            "config_path",
            "dataset_status",
            "ecflow",
            "metadata",
            "platform",
            "reading_chunks",
            "upload",
        ]:
            recipe.pop(k, None)

        metadata["recipe"] = recipe

        metadata["description"] = self.recipe.description
        metadata["licence"] = self.recipe.licence
        metadata["attribution"] = self.recipe.attribution

        metadata["remapping"] = self.output.remapping
        metadata["order_by"] = self.output.order_by
        metadata["flatten_grid"] = self.output.flatten_grid

        metadata["ensemble_dimension"] = len(ensembles)
        metadata["variables"] = variables
        metadata["variables_with_nans"] = variables_with_nans
        metadata["allow_nans"] = self.recipe.build.allow_nans
        metadata["resolution"] = resolution

        metadata["data_request"] = self.minimal_input.data_request
        metadata["field_shape"] = self.minimal_input.field_shape
        metadata["proj_string"] = self.minimal_input.proj_string
        metadata["variables_metadata"] = self.minimal_input.variables_metadata

        metadata["start_date"] = dates[0].isoformat()
        metadata["end_date"] = dates[-1].isoformat()
        metadata["frequency"] = frequency
        metadata["missing_dates"] = [_.isoformat() for _ in missing]

        metadata["version"] = VERSION

        dataset.check_name(
            raise_exception=self.check_name,
            resolution=resolution,
            dates=dates,
            frequency=frequency,
        )

        if len(dates) != total_shape[0]:
            raise ValueError(
                f"Final date size {len(dates)} (from {dates[0]} to {dates[-1]}, {frequency=}) "
                f"does not match data shape {total_shape[0]}. {total_shape=}"
            )

        dates = normalize_and_check_dates(dates, metadata["start_date"], metadata["end_date"], metadata["frequency"])

        # metadata.update(self.recipe.get("force_metadata", {}))

        ###############################################################
        # write metadata
        ###############################################################

        dataset.update_metadata(**metadata)

        dataset.add_dataset(
            name="data",
            chunks=chunks,
            dtype=dtype,
            shape=total_shape,
            dimensions=("time", "variable", "ensemble", "cell"),
        )
        dataset.add_dataset(name="dates", array=dates, dimensions=("time",))
        dataset.add_dataset(name="latitudes", array=grid_points[0], dimensions=("cell",))
        dataset.add_dataset(name="longitudes", array=grid_points[1], dimensions=("cell",))

        statistics_start, statistics_end = build_statistics_dates(
            dates,
            self.recipe.statistics.start,
            self.recipe.statistics.end,
        )
        dataset.update_metadata(statistics_start_date=statistics_start, statistics_end_date=statistics_end)
        LOG.info(f"Will compute statistics from {statistics_start} to {statistics_end}")

        # BACK assert chunks == self.dataset.get_zarr_chunks(), (chunks, self.dataset.get_zarr_chunks())

    def check_unkown_kwargs(self, kwargs: dict) -> None:
        """Check for unknown keyword arguments.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments.
        """
        # remove this latter
        LOG.warning(f"💬 Unknown kwargs for {self.__class__.__name__}: {kwargs}")

    def load_result(self, result: Any, dataset: Dataset) -> None:
        """Load the result into the dataset."""
        # There is one cube to load for each result.
        dates = list(result.group_of_dates)

        LOG.debug(f"Loading cube for {len(dates)} dates")

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

        array = ViewCacheArray(dataset.data, shape=shape, indexes=indexes)
        LOG.info(f"Loading array shape={shape}, indexes={len(indexes)}")
        self.load_cube(cube, array)

        # BACK
        # stats = compute_statistics(array.cache, self.variables_names, allow_nans=self._get_allow_nans())
        # self.tmp_statistics.write(indexes, stats, dates=dates_in_data)
        LOG.info("Flush data array")
        array.flush()
        LOG.info("Flushed data array")

    ######################################################

    def context(self):
        return GriddedContext(
            order_by=self.output.order_by,
            flatten_grid=self.output.flatten_grid,
            remapping=build_remapping(self.output.remapping),
            use_grib_paramid=self.recipe.build.use_grib_paramid,
        )

    def _get_allow_nans(self) -> bool | list:
        """Get the allow_nans configuration.

        Returns
        -------
        bool | list
            The allow_nans configuration.
        """
        config = self.recipe
        if "allow_nans" in config.build:
            return config.build.allow_nans

        return config.statistics.allow_nans

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

    @cached_property
    def allow_nans(self) -> bool | list:
        """Check if NaNs are allowed."""

        z = zarr.open(self.path, mode="r")
        if "allow_nans" in z.attrs:
            return z.attrs["allow_nans"]

        if "variables_with_nans" in z.attrs:
            return z.attrs["variables_with_nans"]

        warnings.warn(f"Cannot find 'variables_with_nans' of 'allow_nans' in {self.path}.")
        return True

    def shape_and_chunks(self, dates: Any) -> tuple[int, ...]:
        """Get the chunks for the dataset."""
        coords = self.minimal_input.coords
        coords["dates"] = dates
        total_shape = list(self.minimal_input.shape)
        total_shape[0] = len(dates)
        LOG.info(f"total_shape = {total_shape}")

        chunks = self.output.get_chunking(coords)
        return total_shape, chunks

    @cached_property
    def variables_names(self) -> list[str]:
        """Get the variable names."""
        z = zarr.open(self.path, mode="r")
        return z.attrs["variables"]
