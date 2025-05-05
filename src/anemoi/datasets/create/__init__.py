# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import os
import time
import uuid
import warnings
from functools import cached_property
from typing import Any
from typing import Optional
from typing import Union

import cftime
import numpy as np
import tqdm
import zarr
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.humanize import compress_dates
from anemoi.utils.humanize import seconds_to_human
from anemoi.utils.sanitise import sanitise
from earthkit.data.core.order import build_remapping

from anemoi.datasets import MissingDateError
from anemoi.datasets import open_dataset
from anemoi.datasets.create.input.trace import enable_trace
from anemoi.datasets.create.persistent import build_storage
from anemoi.datasets.data.misc import as_first_date
from anemoi.datasets.data.misc import as_last_date
from anemoi.datasets.dates.groups import Groups

from .check import DatasetName
from .check import check_data_values
from .chunks import ChunkFilter
from .config import build_output
from .config import loader_config
from .input import build_input
from .statistics import Summary
from .statistics import TmpStatistics
from .statistics import check_variance
from .statistics import compute_statistics
from .statistics import default_statistics_dates
from .statistics import fix_variance
from .utils import normalize_and_check_dates
from .writer import ViewCacheArray

LOG = logging.getLogger(__name__)

VERSION = "0.30"


def json_tidy(o: Any) -> Any:
    """Convert various types to JSON serializable format.

    Parameters
    ----------
    o : Any
        The object to convert.

    Returns
    -------
    Any
        The JSON serializable object.
    """
    if isinstance(o, datetime.datetime):
        return o.isoformat()

    if isinstance(o, datetime.datetime):
        return o.isoformat()

    if isinstance(o, datetime.timedelta):
        return frequency_to_string(o)

    if isinstance(o, cftime.DatetimeJulian):
        import pandas as pd

        o = pd.Timestamp(
            o.year,
            o.month,
            o.day,
            o.hour,
            o.minute,
            o.second,
        )
        return o.isoformat()

    if isinstance(o, (np.float32, np.float64)):
        return float(o)

    raise TypeError(f"{repr(o)} is not JSON serializable {type(o)}")


def build_statistics_dates(
    dates: list[datetime.datetime], start: Optional[datetime.datetime], end: Optional[datetime.datetime]
) -> tuple[str, str]:
    """Compute the start and end dates for the statistics.

    Parameters
    ----------
    dates : list of datetime.datetime
        The list of dates.
    start : Optional[datetime.datetime]
        The start date.
    end : Optional[datetime.datetime]
        The end date.

    Returns
    -------
    tuple of str
        The start and end dates in ISO format.
    """
    # if not specified, use the default statistics dates
    default_start, default_end = default_statistics_dates(dates)
    if start is None:
        start = default_start
    if end is None:
        end = default_end

    # in any case, adapt to the actual dates in the dataset
    start = as_first_date(start, dates)
    end = as_last_date(end, dates)

    # and convert to datetime to isoformat
    start = start.astype(datetime.datetime)
    end = end.astype(datetime.datetime)
    return (start.isoformat(), end.isoformat())


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
    import zarr

    try:
        zarr.open(path, "r")
        return True
    except zarr.errors.PathNotFoundError:
        return False


class Dataset:
    """A class to represent a dataset."""

    def __init__(self, path: str):
        """Initialize a Dataset instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        """
        self.path = path

        _, ext = os.path.splitext(self.path)
        if ext != ".zarr":
            raise ValueError(f"Unsupported extension={ext} for path={self.path}")

    def add_dataset(self, mode: str = "r+", **kwargs: Any) -> zarr.Array:
        """Add a dataset to the Zarr store.

        Parameters
        ----------
        mode : str, optional
            The mode to open the Zarr store.
        **kwargs
            Additional arguments for the dataset.

        Returns
        -------
        zarr.Array
            The added dataset.
        """
        import zarr

        z = zarr.open(self.path, mode=mode)
        from .zarr import add_zarr_dataset

        return add_zarr_dataset(zarr_root=z, **kwargs)

    def update_metadata(self, **kwargs: Any) -> None:
        """Update the metadata of the dataset.

        Parameters
        ----------
        **kwargs
            The metadata to update.
        """
        import zarr

        LOG.debug(f"Updating metadata {kwargs}")
        z = zarr.open(self.path, mode="w+")
        for k, v in kwargs.items():
            if isinstance(v, np.datetime64):
                v = v.astype(datetime.datetime)
            if isinstance(v, datetime.date):
                v = v.isoformat()
            z.attrs[k] = json.loads(json.dumps(v, default=json_tidy))

    @cached_property
    def anemoi_dataset(self) -> Any:
        """Get the Anemoi dataset."""
        return open_dataset(self.path)

    @cached_property
    def zarr_metadata(self) -> dict:
        """Get the Zarr metadata."""
        import zarr

        return dict(zarr.open(self.path, mode="r").attrs)

    def print_info(self) -> None:
        """Print information about the dataset."""
        import zarr

        z = zarr.open(self.path, mode="r")
        try:
            LOG.info(z["data"].info)
        except Exception as e:
            LOG.info(e)

    def get_zarr_chunks(self) -> tuple:
        """Get the chunks of the Zarr dataset.

        Returns
        -------
        tuple
            The chunks of the Zarr dataset.
        """
        import zarr

        z = zarr.open(self.path, mode="r")
        return z["data"].chunks

    def check_name(
        self,
        resolution: str,
        dates: list[datetime.datetime],
        frequency: datetime.timedelta,
        raise_exception: bool = True,
        is_test: bool = False,
    ) -> None:
        """Check the name of the dataset.

        Parameters
        ----------
        resolution : str
            The resolution of the dataset.
        dates : list of datetime.datetime
            The dates of the dataset.
        frequency : datetime.timedelta
            The frequency of the dataset.
        raise_exception : bool, optional
            Whether to raise an exception if the name is invalid.
        is_test : bool, optional
            Whether this is a test.
        """
        basename, _ = os.path.splitext(os.path.basename(self.path))
        try:
            DatasetName(basename, resolution, dates[0], dates[-1], frequency).raise_if_not_valid()
        except Exception as e:
            if raise_exception and not is_test:
                raise e
            else:
                LOG.warning(f"Dataset name error: {e}")

    def get_main_config(self) -> Any:
        """Get the main configuration of the dataset.

        Returns
        -------
        Any
            The main configuration.
        """
        import zarr

        z = zarr.open(self.path, mode="r")
        config = loader_config(z.attrs.get("_create_yaml_config"))

        if "env" in config:
            for k, v in config["env"].items():
                LOG.info(f"Setting env variable {k}={v}")
                os.environ[k] = str(v)

        return config


class WritableDataset(Dataset):
    """A class to represent a writable dataset."""

    def __init__(self, path: str):
        """Initialize a WritableDataset instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        """
        super().__init__(path)
        self.path = path

        import zarr

        self.z = zarr.open(self.path, mode="r+")

    @cached_property
    def data_array(self) -> Any:
        """Get the data array of the dataset."""
        import zarr

        return zarr.open(self.path, mode="r+")["data"]


class NewDataset(Dataset):
    """A class to represent a new dataset."""

    def __init__(self, path: str, overwrite: bool = False):
        """Initialize a NewDataset instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        overwrite : bool, optional
            Whether to overwrite the existing dataset.
        """
        super().__init__(path)
        self.path = path

        import zarr

        self.z = zarr.open(self.path, mode="w")
        self.z.create_group("_build")


class Actor:  # TODO: rename to Creator
    """A base class for dataset creation actors."""

    dataset_class = WritableDataset

    def __init__(self, path: str, cache: Optional[str] = None):
        """Initialize an Actor instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        cache : Optional[str], optional
            The cache directory.
        """
        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise", under="warn")

        self.path = path
        self.cache = cache
        self.dataset = self.dataset_class(self.path)

    def run(self) -> None:
        """Run the actor."""
        # to be implemented in the sub-classes
        raise NotImplementedError()

    def update_metadata(self, **kwargs: Any) -> None:
        """Update the metadata of the dataset.

        Parameters
        ----------
        **kwargs
            The metadata to update.
        """
        self.dataset.update_metadata(**kwargs)

    def _cache_context(self) -> Any:
        """Get the cache context.

        Returns
        -------
        Any
            The cache context.
        """
        from .utils import cache_context

        return cache_context(self.cache)

    def check_unkown_kwargs(self, kwargs: dict) -> None:
        """Check for unknown keyword arguments.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments.
        """
        # remove this latter
        LOG.warning(f"💬 Unknown kwargs for {self.__class__.__name__}: {kwargs}")

    def read_dataset_metadata(self, path: str) -> None:
        """Read the metadata of the dataset.

        Parameters
        ----------
        path : str
            The path to the dataset.
        """
        ds = open_dataset(path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables
        assert len(self.variables_names) == ds.shape[1], self.dataset_shape
        self.dates = ds.dates

        self.missing_dates = sorted(list([self.dates[i] for i in ds.missing]))

        def check_missing_dates(expected: list[np.datetime64]) -> None:
            """Check if the missing dates in the dataset match the expected dates.

            Parameters
            ----------
            expected : list of np.datetime64
                The expected missing dates.

            Raises
            ------
            ValueError
                If the missing dates in the dataset do not match the expected dates.
            """
            import zarr

            z = zarr.open(path, "r")
            missing_dates = z.attrs.get("missing_dates", [])
            missing_dates = sorted([np.datetime64(d) for d in missing_dates])
            if missing_dates != expected:
                LOG.warning("Missing dates given in recipe do not match the actual missing dates in the dataset.")
                LOG.warning(f"Missing dates in recipe: {sorted(str(x) for x in missing_dates)}")
                LOG.warning(f"Missing dates in dataset: {sorted(str(x) for x in  expected)}")
                raise ValueError("Missing dates given in recipe do not match the actual missing dates in the dataset.")

        check_missing_dates(self.missing_dates)


class Patch(Actor):
    """A class to apply patches to a dataset."""

    def __init__(self, path: str, options: dict = None, **kwargs: Any):
        """Initialize a Patch instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        options : dict, optional
            The patch options.
        """
        self.path = path
        self.options = options or {}

    def run(self) -> None:
        """Run the patch."""
        from .patch import apply_patch

        apply_patch(self.path, **self.options)


class Size(Actor):
    """A class to compute the size of a dataset."""

    def __init__(self, path: str, **kwargs: Any):
        """Initialize a Size instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        """
        super().__init__(path)

    def run(self) -> None:
        """Run the size computation."""
        from .size import compute_directory_sizes

        metadata = compute_directory_sizes(self.path)
        self.update_metadata(**metadata)

        # Look for constant fields
        ds = open_dataset(self.path)
        constants = ds.computed_constant_fields()

        variables_metadata = self.dataset.zarr_metadata.get("variables_metadata", {}).copy()
        for k in constants:
            variables_metadata[k]["constant_in_time"] = True

        self.update_metadata(constant_fields=constants, variables_metadata=variables_metadata)


class HasRegistryMixin:
    """A mixin class to provide registry functionality."""

    @cached_property
    def registry(self) -> Any:
        """Get the registry."""
        from .zarr import ZarrBuiltRegistry

        return ZarrBuiltRegistry(self.path, use_threads=self.use_threads)


class HasStatisticTempMixin:
    """A mixin class to provide temporary statistics functionality."""

    @cached_property
    def tmp_statistics(self) -> TmpStatistics:
        """Get the temporary statistics."""
        directory = self.statistics_temp_dir or os.path.join(self.path + ".storage_for_statistics.tmp")
        return TmpStatistics(directory)


class HasElementForDataMixin:
    """A mixin class to provide element creation functionality for data."""

    def create_elements(self, config: Any) -> None:
        """Create elements for the dataset.

        Parameters
        ----------
        config : Any
            The configuration.
        """
        assert self.registry
        assert self.tmp_statistics

        LOG.info(dict(config.dates))

        self.groups = Groups(**config.dates)
        LOG.info(self.groups)

        self.output = build_output(config.output, parent=self)

        self.input = build_input_(main_config=config, output_config=self.output)
        # LOG.info("%s", self.input)


def build_input_(main_config: Any, output_config: Any) -> Any:
    """Build the input for the dataset.

    Parameters
    ----------
    main_config : Any
        The main configuration.
    output_config : Any
        The output configuration.

    Returns
    -------
    Any
        The input builder.
    """
    builder = build_input(
        main_config.input,
        data_sources=main_config.get("data_sources", {}),
        order_by=output_config.order_by,
        flatten_grid=output_config.flatten_grid,
        remapping=build_remapping(output_config.remapping),
        use_grib_paramid=main_config.build.use_grib_paramid,
    )
    LOG.debug("✅ INPUT_BUILDER")
    LOG.debug(builder)
    return builder


class Init(Actor, HasRegistryMixin, HasStatisticTempMixin, HasElementForDataMixin):
    """A class to initialize a new dataset."""

    dataset_class = NewDataset

    def __init__(
        self,
        path: str,
        config: dict,
        check_name: bool = False,
        overwrite: bool = False,
        use_threads: bool = False,
        statistics_temp_dir: Optional[str] = None,
        progress: Any = None,
        test: bool = False,
        cache: Optional[str] = None,
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
        if _path_readable(path) and not overwrite:
            raise Exception(f"{path} already exists. Use overwrite=True to overwrite.")

        super().__init__(path, cache=cache)
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

        one_date = self.groups.one_date()
        # assert False, (type(one_date), type(self.groups))
        self.minimal_input = self.input.select(one_date)
        LOG.info(f"Minimal input for 'init' step (using only the first date) : {one_date}")
        LOG.info(self.minimal_input)

    def run(self) -> int:
        """Run the initialization.

        Returns
        -------
        int
            The number of groups to process.
        """
        with self._cache_context():
            return self._run()

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

        statistics_start, statistics_end = build_statistics_dates(
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


class Load(Actor, HasRegistryMixin, HasStatisticTempMixin, HasElementForDataMixin):
    """A class to load data into a dataset."""

    def __init__(
        self,
        path: str,
        parts: Optional[str] = None,
        use_threads: bool = False,
        statistics_temp_dir: Optional[str] = None,
        progress: Any = None,
        cache: Optional[str] = None,
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
        super().__init__(path, cache=cache)
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

    def run(self) -> None:
        """Run the data loading."""
        with self._cache_context():
            self._run()

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

            result = self.input.select(group_of_dates=group)
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

                a = set(as_datetime(_) for _ in dates)
                b = set(as_datetime(_) for _ in dates_in_data)

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

    def _get_allow_nans(self) -> Union[bool, list]:
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

        def position(x: Any) -> Optional[int]:
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


class Cleanup(Actor, HasRegistryMixin, HasStatisticTempMixin):
    """A class to clean up temporary data and registry entries."""

    def __init__(
        self,
        path: str,
        statistics_temp_dir: Optional[str] = None,
        delta: list = [],
        use_threads: bool = False,
        **kwargs: Any,
    ):
        """Initialize a Cleanup instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        statistics_temp_dir : Optional[str], optional
            The directory for temporary statistics.
        delta : list, optional
            The delta values.
        use_threads : bool, optional
            Whether to use threads.
        """
        super().__init__(path)
        self.use_threads = use_threads
        self.statistics_temp_dir = statistics_temp_dir
        self.additinon_temp_dir = statistics_temp_dir
        self.actors = [
            _InitAdditions(path, delta=d, use_threads=use_threads, statistics_temp_dir=statistics_temp_dir)
            for d in delta
        ]

    def run(self) -> None:
        """Run the cleanup."""

        self.tmp_statistics.delete()
        self.registry.clean()
        for actor in self.actors:
            actor.cleanup()


class Verify(Actor):
    """A class to verify the integrity of a dataset."""

    def __init__(self, path: str, **kwargs: Any):
        """Initialize a Verify instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        """
        super().__init__(path)

    def run(self) -> None:
        """Run the verification."""
        LOG.info(f"Verifying dataset at {self.path}")
        LOG.info(str(self.dataset.anemoi_dataset))


class AdditionsMixin:
    """A mixin class to handle dataset additions."""

    def skip(self) -> bool:
        """Check if the additions should be skipped.

        Returns
        -------
        bool
            Whether to skip the additions.
        """
        frequency = frequency_to_timedelta(self.dataset.anemoi_dataset.frequency)
        if not self.delta.total_seconds() % frequency.total_seconds() == 0:
            LOG.debug(f"Delta {self.delta} is not a multiple of frequency {frequency}. Skipping.")
            return True

        if self.dataset.zarr_metadata.get("build", {}).get("additions", None) is False:
            LOG.warning(f"Additions are disabled for {self.path} in the recipe.")
            return True

        return False

    @cached_property
    def tmp_storage_path(self) -> str:
        """Get the path to the temporary storage."""
        name = "storage_for_additions"
        if self.delta:
            name += frequency_to_string(self.delta)
        return os.path.join(f"{self.path}.{name}.tmp")

    def read_from_dataset(self) -> None:
        """Read data from the dataset."""
        self.variables = self.dataset.anemoi_dataset.variables
        self.frequency = frequency_to_timedelta(self.dataset.anemoi_dataset.frequency)
        start = self.dataset.zarr_metadata["statistics_start_date"]
        end = self.dataset.zarr_metadata["statistics_end_date"]
        self.start = datetime.datetime.fromisoformat(start)
        self.end = datetime.datetime.fromisoformat(end)

        ds = open_dataset(self.path, start=self.start, end=self.end)
        self.dates = ds.dates
        self.total = len(self.dates)

        idelta = self.delta.total_seconds() // self.frequency.total_seconds()
        assert int(idelta) == idelta, idelta
        idelta = int(idelta)
        self.ds = DeltaDataset(ds, idelta)


class DeltaDataset:
    """A class to represent a dataset with delta values."""

    def __init__(self, ds: Any, idelta: int):
        """Initialize a DeltaDataset instance.

        Parameters
        ----------
        ds : Any
            The dataset.
        idelta : int
            The delta value.
        """
        self.ds = ds
        self.idelta = idelta

    def __getitem__(self, i: int) -> Any:
        """Get an item from the dataset.

        Parameters
        ----------
        i : int
            The index.

        Returns
        -------
        Any
            The item.
        """
        j = i - self.idelta
        if j < 0:
            raise MissingDateError(f"Missing date {j}")
        return self.ds[i : i + 1, ...] - self.ds[j : j + 1, ...]


class _InitAdditions(Actor, HasRegistryMixin, AdditionsMixin):
    """A class to initialize dataset additions."""

    def __init__(self, path: str, delta: str, use_threads: bool = False, progress: Any = None, **kwargs: Any):
        """Initialize an _InitAdditions instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        delta : str
            The delta value.
        use_threads : bool, optional
            Whether to use threads.
        progress : Any, optional
            The progress indicator.
        """
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress

    def run(self) -> None:
        """Run the additions initialization."""
        if self.skip():
            LOG.info(f"Skipping delta={self.delta}")
            return

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=True)
        self.tmp_storage.delete()
        self.tmp_storage.create()
        LOG.info(f"Dataset {self.tmp_storage_path} additions initialised.")

    def cleanup(self) -> None:
        """Clean up the temporary storage."""
        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        self.tmp_storage.delete()
        LOG.info(f"Cleaned temporary storage {self.tmp_storage_path}")


class _RunAdditions(Actor, HasRegistryMixin, AdditionsMixin):
    """A class to run dataset additions."""

    def __init__(
        self,
        path: str,
        delta: str,
        parts: Optional[str] = None,
        use_threads: bool = False,
        progress: Any = None,
        **kwargs: Any,
    ):
        """Initialize a _RunAdditions instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        delta : str
            The delta value.
        parts : Optional[str], optional
            The parts to load.
        use_threads : bool, optional
            Whether to use threads.
        progress : Any, optional
            The progress indicator.
        """
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress
        self.parts = parts

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        LOG.info(f"Writing in {self.tmp_storage_path}")

    def run(self) -> None:
        """Run the additions."""
        if self.skip():
            LOG.info(f"Skipping delta={self.delta}")
            return

        self.read_from_dataset()

        chunk_filter = ChunkFilter(parts=self.parts, total=self.total)
        for i in range(0, self.total):
            if not chunk_filter(i):
                continue
            date = self.dates[i]
            try:
                arr = self.ds[i]
                stats = compute_statistics(arr, self.variables, allow_nans=self.allow_nans)
                self.tmp_storage.add([date, i, stats], key=date)
            except MissingDateError:
                self.tmp_storage.add([date, i, "missing"], key=date)
        self.tmp_storage.flush()
        LOG.debug(f"Dataset {self.path} additions run.")

    def allow_nans(self) -> bool:
        """Check if NaNs are allowed.

        Returns
        -------
        bool
            Whether NaNs are allowed.
        """
        if self.dataset.anemoi_dataset.metadata.get("allow_nans", False):
            return True

        variables_with_nans = self.dataset.anemoi_dataset.metadata.get("variables_with_nans", None)
        if variables_with_nans is not None:
            return variables_with_nans
        warnings.warn(f"❗Cannot find 'variables_with_nans' in {self.path}, assuming nans allowed.")
        return True


class _FinaliseAdditions(Actor, HasRegistryMixin, AdditionsMixin):
    """A class to finalize dataset additions."""

    def __init__(self, path: str, delta: str, use_threads: bool = False, progress: Any = None, **kwargs: Any):
        """Initialize a _FinaliseAdditions instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        delta : str
            The delta value.
        use_threads : bool, optional
            Whether to use threads.
        progress : Any, optional
            The progress indicator.
        """
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        LOG.info(f"Reading from {self.tmp_storage_path}.")

    def run(self) -> None:
        """Run the additions finalization."""
        if self.skip():
            LOG.info(f"Skipping delta={self.delta}.")
            return

        self.read_from_dataset()

        shape = (len(self.dates), len(self.variables))
        agg = dict(
            minimum=np.full(shape, np.nan, dtype=np.float64),
            maximum=np.full(shape, np.nan, dtype=np.float64),
            sums=np.full(shape, np.nan, dtype=np.float64),
            squares=np.full(shape, np.nan, dtype=np.float64),
            count=np.full(shape, -1, dtype=np.int64),
            has_nans=np.full(shape, False, dtype=np.bool_),
        )
        LOG.debug(f"Aggregating {self.__class__.__name__} statistics on shape={shape}. Variables : {self.variables}")

        found = set()
        ifound = set()
        missing = set()
        for _date, (date, i, stats) in self.tmp_storage.items():
            assert _date == date
            if stats == "missing":
                missing.add(date)
                continue

            assert date not in found, f"Duplicates found {date}"
            found.add(date)
            ifound.add(i)

            for k in ["minimum", "maximum", "sums", "squares", "count", "has_nans"]:
                agg[k][i, ...] = stats[k]

        assert len(found) + len(missing) == len(self.dates), (
            len(found),
            len(missing),
            len(self.dates),
        )
        assert found.union(missing) == set(self.dates), (
            found,
            missing,
            set(self.dates),
        )

        if len(ifound) < 2:
            LOG.warning(f"Not enough data found in {self.path} to compute {self.__class__.__name__}. Skipped.")
            self.tmp_storage.delete()
            return

        mask = sorted(list(ifound))
        for k in ["minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            agg[k] = agg[k][mask, ...]

        for k in ["minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            assert agg[k].shape == agg["count"].shape, (
                agg[k].shape,
                agg["count"].shape,
            )

        minimum = np.nanmin(agg["minimum"], axis=0)
        maximum = np.nanmax(agg["maximum"], axis=0)
        sums = np.nansum(agg["sums"], axis=0)
        squares = np.nansum(agg["squares"], axis=0)
        count = np.nansum(agg["count"], axis=0)
        has_nans = np.any(agg["has_nans"], axis=0)

        assert sums.shape == count.shape
        assert sums.shape == squares.shape
        assert sums.shape == minimum.shape
        assert sums.shape == maximum.shape
        assert sums.shape == has_nans.shape

        mean = sums / count
        assert sums.shape == mean.shape

        x = squares / count - mean * mean
        # x[- 1e-15 < (x / (np.sqrt(squares / count) + np.abs(mean))) < 0] = 0
        # remove negative variance due to numerical errors
        for i, name in enumerate(self.variables):
            x[i] = fix_variance(x[i], name, agg["count"][i : i + 1], agg["sums"][i : i + 1], agg["squares"][i : i + 1])
        check_variance(x, self.variables, minimum, maximum, mean, count, sums, squares)

        stdev = np.sqrt(x)
        assert sums.shape == stdev.shape

        self.summary = Summary(
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            count=count,
            sums=sums,
            squares=squares,
            stdev=stdev,
            variables_names=self.variables,
            has_nans=has_nans,
        )
        LOG.info(f"Dataset {self.path} additions finalised.")
        # self.check_statistics()
        self._write(self.summary)
        self.tmp_storage.delete()

    def _write(self, summary: Summary) -> None:
        """Write the summary to the dataset.

        Parameters
        ----------
        summary : Summary
            The summary to write.
        """
        for k in ["mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            name = f"statistics_tendencies_{frequency_to_string(self.delta)}_{k}"
            self.dataset.add_dataset(name=name, array=summary[k], dimensions=("variable",))
        self.registry.add_to_history(f"compute_statistics_{self.__class__.__name__.lower()}_end")
        LOG.debug(f"Wrote additions in {self.path}")


def multi_addition(cls: type) -> type:
    """Create a class to handle multiple additions.

    Parameters
    ----------
    cls : type
        The class to handle additions.

    Returns
    -------
    type
        The class to handle multiple additions.
    """

    class MultiAdditions:
        def __init__(self, *args, **kwargs: Any):
            self.actors = []

            for k in kwargs.pop("delta", []):
                self.actors.append(cls(*args, delta=k, **kwargs))

            if not self.actors:
                LOG.warning("No delta found in kwargs, no additions will be computed.")

        def run(self) -> None:
            """Run the additions."""
            for actor in self.actors:
                actor.run()

    return MultiAdditions


InitAdditions = multi_addition(_InitAdditions)
RunAdditions = multi_addition(_RunAdditions)
FinaliseAdditions = multi_addition(_FinaliseAdditions)


class Statistics(Actor, HasStatisticTempMixin, HasRegistryMixin):
    """A class to compute statistics for a dataset."""

    def __init__(
        self,
        path: str,
        use_threads: bool = False,
        statistics_temp_dir: Optional[str] = None,
        progress: Any = None,
        **kwargs: Any,
    ):
        """Initialize a Statistics instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        use_threads : bool, optional
            Whether to use threads.
        statistics_temp_dir : Optional[str], optional
            The directory for temporary statistics.
        progress : Any, optional
            The progress indicator.
        """
        super().__init__(path)
        self.use_threads = use_threads
        self.progress = progress
        self.statistics_temp_dir = statistics_temp_dir

    def run(self) -> None:
        """Run the statistics computation."""
        start, end = (
            self.dataset.zarr_metadata["statistics_start_date"],
            self.dataset.zarr_metadata["statistics_end_date"],
        )
        start, end = np.datetime64(start), np.datetime64(end)
        dates = self.dataset.anemoi_dataset.dates

        assert type(dates[0]) is type(start), (type(dates[0]), type(start))

        dates = [d for d in dates if d >= start and d <= end]
        dates = [d for i, d in enumerate(dates) if i not in self.dataset.anemoi_dataset.missing]
        variables = self.dataset.anemoi_dataset.variables
        stats = self.tmp_statistics.get_aggregated(dates, variables, self.allow_nans)

        LOG.info(stats)

        if not all(self.registry.get_flags(sync=False)):
            raise Exception(f"❗Zarr {self.path} is not fully built, not writing statistics into dataset.")

        for k in ["mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            self.dataset.add_dataset(name=k, array=stats[k], dimensions=("variable",))

        self.registry.add_to_history("compute_statistics_end")
        LOG.info(f"Wrote statistics in {self.path}")

    @cached_property
    def allow_nans(self) -> Union[bool, list]:
        """Check if NaNs are allowed."""
        import zarr

        z = zarr.open(self.path, mode="r")
        if "allow_nans" in z.attrs:
            return z.attrs["allow_nans"]

        if "variables_with_nans" in z.attrs:
            return z.attrs["variables_with_nans"]

        warnings.warn(f"Cannot find 'variables_with_nans' of 'allow_nans' in {self.path}.")
        return True


def chain(tasks: list) -> type:
    """Create a class to chain multiple tasks.

    Parameters
    ----------
    tasks : list
        The list of tasks to chain.

    Returns
    -------
    type
        The class to chain multiple tasks.
    """

    class Chain(Actor):
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

        def run(self) -> None:
            """Run the chained tasks."""
            for cls in tasks:
                t = cls(**self.kwargs)
                t.run()

    return Chain


def creator_factory(name: str, trace: Optional[str] = None, **kwargs: Any) -> Any:
    """Create a dataset creator.

    Parameters
    ----------
    name : str
        The name of the creator.
    trace : Optional[str], optional
        The trace file.
    **kwargs
        Additional arguments for the creator.

    Returns
    -------
    Any
        The dataset creator.
    """
    if trace:

        enable_trace(trace)

    cls = dict(
        init=Init,
        load=Load,
        size=Size,
        patch=Patch,
        statistics=Statistics,
        finalise=chain([Statistics, Size, Cleanup]),
        cleanup=Cleanup,
        verify=Verify,
        init_additions=InitAdditions,
        load_additions=RunAdditions,
        run_additions=RunAdditions,
        finalise_additions=chain([FinaliseAdditions, Size]),
        additions=chain([InitAdditions, RunAdditions, FinaliseAdditions, Size, Cleanup]),
    )[name]
    LOG.debug(f"Creating {cls.__name__} with {kwargs}")
    return cls(**kwargs)
