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
import warnings
from functools import cached_property
from typing import Any

import numpy as np
import tqdm
import zarr
from anemoi.utils.dates import as_datetime
from anemoi.utils.humanize import compress_dates
from anemoi.utils.humanize import seconds_to_human

from ..check import check_data_values
from ..creator import Creator
from ..dataset import Dataset
from .context import GriddedContext
from .writer import ViewCacheArray

LOG = logging.getLogger(__name__)


class GriddedCreator(Creator):

    check_name = False

    def collect_metadata(self, metadata: dict):
        """Run the initialisation process for the dataset."""
        super().collect_metadata(metadata)

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

    def initialise_dataset(self, dataset: Dataset) -> None:
        super().initialise_dataset(dataset)

        dates = self.groups.provider.values
        total_shape = (len(dates),) + self.minimal_input.shape[1:]

        assert len(total_shape) == 4, f"Expected 4D shape, got {total_shape}"

        coords = self.minimal_input.coords
        coords["dates"] = dates
        chunks = self.recipe.output.get_chunking(coords)

        grid_points = self.minimal_input.grid_points

        # Create arrays

        dataset.add_array(
            name="data",
            chunks=chunks,
            dtype=self.recipe.output.dtype,
            shape=total_shape,
            dimensions=("time", "variable", "ensemble", "cell"),
            fill_value=np.nan,
        )

        dataset.add_array(name="dates", data=np.array(dates, "<M8[s]"), dimensions=("time",))
        dataset.add_array(name="latitudes", data=grid_points[0], dimensions=("cell",))
        dataset.add_array(name="longitudes", data=grid_points[1], dimensions=("cell",))

    def xxxxxcheck_unkown_kwargs(self, kwargs: dict) -> None:
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

        array = ViewCacheArray(dataset.data, shape=shape, indexes=indexes)
        LOG.info(f"Loading array shape={shape}, indexes={len(indexes)}")
        self.load_cube(cube, array)

        LOG.info("Flush data array")
        array.flush()
        LOG.info("Flushed data array")

    def check_dataset_name(self, path: str) -> None:
        LOG.warning("BACK: Dataset name checking not yes implemented.")

    ######################################################

    def context(self):
        return GriddedContext(self.recipe)

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
