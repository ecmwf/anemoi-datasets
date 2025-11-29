# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import uuid
from typing import Any

import zarr
from anemoi.utils.sanitise import sanitise

from anemoi.datasets.create.base.init import InitTask
from anemoi.datasets.create.config import loader_config
from anemoi.datasets.create.utils import normalize_and_check_dates

from .tasks import GriddedTaskMixin
from .tasks import NewDataset
from .tasks import _build_statistics_dates

LOG = logging.getLogger(__name__)

VERSION = "0.30"


def _path_readable(path: str) -> bool:
    """Check if the path is readable.

    Parameters
    ----------
    path : str
        The path to check.

    Returns
    -------
    bool
        True if the path is readable, False otherwise.
    """

    try:
        zarr.open(path, "r")
        return True
    except zarr.errors.PathNotFoundError:
        return False


class Init(InitTask, GriddedTaskMixin):
    """A class to initialize a new dataset."""

    dataset_class = NewDataset

    def __init__(
        self,
        path: str,
        config: dict,
        check_name: bool = False,
        overwrite: bool = False,
        use_threads: bool = False,
        statistics_temp_dir: str | None = None,
        progress: Any = None,
        test: bool = False,
        cache: str | None = None,
        **kwargs: Any,
    ):
        """Initialize an Init instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        config : dict
            The configuration.
        check_name : bool, optional
            Whether to check the dataset name.
        overwrite : bool, optional
            Whether to overwrite the existing dataset.
        use_threads : bool, optional
            Whether to use threads.
        statistics_temp_dir : Optional[str], optional
            The directory for temporary statistics.
        progress : Any, optional
            The progress indicator.
        test : bool, optional
            Whether this is a test.
        cache : Optional[str], optional
            The cache directory.
        """
        super().__init__(path, config, overwrite=overwrite, cache=cache)
        if _path_readable(path) and not overwrite:
            raise Exception(f"{path} already exists. Use overwrite=True to overwrite.")

        self.config = config
        self.check_name = check_name
        self.use_threads = use_threads
        self.statistics_temp_dir = statistics_temp_dir
        self.progress = progress
        self.test = test

        self.main_config = loader_config(config, is_test=test)

        # self.registry.delete() ??
        self.tmp_statistics.delete()

        assert isinstance(self.main_config.output.order_by, dict), self.main_config.output.order_by
        self.create_elements(self.main_config)

        LOG.info(f"Groups: {self.groups}")

        # window = self.main_config.dates.get("window")

        one_date = self.groups.one_date()

        self.minimal_input = self.input.select(self.context, one_date)

        LOG.info(f"Minimal input for 'init' step (using only the first date) : {one_date}")
        LOG.info(self.minimal_input)

    def _run(self) -> int:
        """Internal method to run the initialization.

        Returns
        -------
        int
            The number of groups to process.
        """
        """Create an empty dataset of the right final shape.

        Read a small part of the data to get the shape of the data and the resolution and more metadata.
        """

        LOG.info("Config loaded ok:")
        # LOG.info(self.main_config)

        dates = self.groups.provider.values
        frequency = self.groups.provider.frequency
        missing = self.groups.provider.missing

        assert isinstance(frequency, datetime.timedelta), frequency

        LOG.info(f"Found {len(dates)} datetimes.")
        LOG.info(f"Dates: Found {len(dates)} datetimes, in {len(self.groups)} groups: ")
        LOG.info(f"Missing dates: {len(missing)}")
        lengths = tuple(len(g) for g in self.groups)

        variables = self.minimal_input.variables
        LOG.info(f"Found {len(variables)} variables : {','.join(variables)}.")

        variables_with_nans = self.main_config.statistics.get("allow_nans", [])

        ensembles = self.minimal_input.ensembles
        LOG.info(f"Found {len(ensembles)} ensembles : {','.join([str(_) for _ in ensembles])}.")

        grid_points = self.minimal_input.grid_points
        LOG.info(f"gridpoints size: {[len(i) for i in grid_points]}")

        resolution = self.minimal_input.resolution
        LOG.info(f"{resolution=}")

        coords = self.minimal_input.coords
        coords["dates"] = dates
        total_shape = self.minimal_input.shape
        total_shape[0] = len(dates)
        LOG.info(f"total_shape = {total_shape}")

        chunks = self.output.get_chunking(coords)
        LOG.info(f"{chunks=}")
        dtype = self.output.dtype

        LOG.info(f"Creating Dataset '{self.path}', with {total_shape=}, {chunks=} and {dtype=}")

        metadata = {}
        metadata["uuid"] = str(uuid.uuid4())

        metadata.update(self.main_config.get("add_metadata", {}))

        metadata["_create_yaml_config"] = self.main_config.get_serialisable_dict()

        recipe = sanitise(self.main_config.get_serialisable_dict())

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

        metadata["description"] = self.main_config.description
        metadata["licence"] = self.main_config["licence"]
        metadata["attribution"] = self.main_config["attribution"]

        metadata["remapping"] = self.output.remapping
        metadata["order_by"] = self.output.order_by_as_list
        metadata["flatten_grid"] = self.output.flatten_grid

        metadata["ensemble_dimension"] = len(ensembles)
        metadata["variables"] = variables
        metadata["variables_with_nans"] = variables_with_nans
        metadata["allow_nans"] = self.main_config.build.get("allow_nans", False)
        metadata["resolution"] = resolution

        metadata["data_request"] = self.minimal_input.data_request
        metadata["field_shape"] = self.minimal_input.field_shape
        metadata["proj_string"] = self.minimal_input.proj_string
        metadata["variables_metadata"] = self.minimal_input.variables_metadata

        metadata["start_date"] = dates[0].isoformat()
        metadata["end_date"] = dates[-1].isoformat()
        metadata["frequency"] = frequency
        metadata["missing_dates"] = [_.isoformat() for _ in missing]
        metadata["origins"] = self.minimal_input.origins

        metadata["version"] = VERSION

        self.dataset.check_name(
            raise_exception=self.check_name,
            is_test=self.test,
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

        metadata.update(self.main_config.get("force_metadata", {}))

        ###############################################################
        # write metadata
        ###############################################################

        self.update_metadata(**metadata)

        self.dataset.add_dataset(
            name="data",
            chunks=chunks,
            dtype=dtype,
            shape=total_shape,
            dimensions=("time", "variable", "ensemble", "cell"),
        )
        self.dataset.add_dataset(name="dates", array=dates, dimensions=("time",))
        self.dataset.add_dataset(name="latitudes", array=grid_points[0], dimensions=("cell",))
        self.dataset.add_dataset(name="longitudes", array=grid_points[1], dimensions=("cell",))

        self.registry.create(lengths=lengths)
        self.tmp_statistics.create(exist_ok=False)
        self.registry.add_to_history("tmp_statistics_initialised", version=self.tmp_statistics.version)

        statistics_start, statistics_end = _build_statistics_dates(
            dates,
            self.main_config.statistics.get("start"),
            self.main_config.statistics.get("end"),
        )
        self.update_metadata(statistics_start_date=statistics_start, statistics_end_date=statistics_end)
        LOG.info(f"Will compute statistics from {statistics_start} to {statistics_end}")

        self.registry.add_to_history("init finished")

        assert chunks == self.dataset.get_zarr_chunks(), (chunks, self.dataset.get_zarr_chunks())

        # Return the number of groups to process, so we can show a nice progress bar
        return len(lengths)
