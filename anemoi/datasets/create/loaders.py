# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import datetime
import logging
import os
import time
import uuid
from functools import cached_property

import numpy as np
import zarr

from anemoi.datasets import open_dataset
from anemoi.datasets.data.misc import as_first_date
from anemoi.datasets.data.misc import as_last_date
from anemoi.datasets.dates.groups import Groups

from .check import DatasetName
from .check import check_data_values
from .config import build_output
from .config import loader_config
from .input import build_input
from .statistics import TempStatistics
from .statistics import compute_statistics
from .utils import bytes
from .utils import compute_directory_sizes
from .utils import normalize_and_check_dates
from .utils import progress_bar
from .utils import seconds
from .writer import CubesFilter
from .writer import ViewCacheArray
from .zarr import ZarrBuiltRegistry
from .zarr import add_zarr_dataset

LOG = logging.getLogger(__name__)

VERSION = "0.20"


def default_statistics_dates(dates):
    """
    Calculate default statistics dates based on the given list of dates.

    Args:
        dates (list): List of datetime objects representing dates.

    Returns:
        tuple: A tuple containing the default start and end dates.
    """

    def to_datetime(d):
        if isinstance(d, np.datetime64):
            return d.tolist()
        assert isinstance(d, datetime.datetime), d
        return d

    first = dates[0]
    last = dates[-1]

    first = to_datetime(first)
    last = to_datetime(last)

    n_years = round((last - first).total_seconds() / (365.25 * 24 * 60 * 60))

    if n_years < 10:
        # leave out 20% of the data
        k = int(len(dates) * 0.8)
        end = dates[k - 1]
        LOG.info(f"Number of years {n_years} < 10, leaving out 20%. {end=}")
        return dates[0], end

    delta = 1
    if n_years >= 20:
        delta = 3
    LOG.info(f"Number of years {n_years}, leaving out {delta} years.")
    end_year = last.year - delta

    end = max(d for d in dates if to_datetime(d).year == end_year)
    return dates[0], end


class Loader:
    def __init__(self, *, path, print=print, **kwargs):
        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise", under="warn")

        assert isinstance(path, str), path

        self.path = path
        self.kwargs = kwargs
        self.print = print

        statistics_tmp = kwargs.get("statistics_tmp") or self.path + ".statistics"

        self.statistics_registry = TempStatistics(statistics_tmp)

    @classmethod
    def from_config(cls, *, config, path, print=print, **kwargs):
        # config is the path to the config file or a dict with the config
        assert isinstance(config, dict) or isinstance(config, str), config
        return cls(config=config, path=path, print=print, **kwargs)

    @classmethod
    def from_dataset_config(cls, *, path, print=print, **kwargs):
        assert os.path.exists(path), f"Path {path} does not exist."
        z = zarr.open(path, mode="r")
        config = z.attrs["_create_yaml_config"]
        LOG.info(f"Config loaded from zarr config: {config}")
        return cls.from_config(config=config, path=path, print=print, **kwargs)

    @classmethod
    def from_dataset(cls, *, path, **kwargs):
        assert os.path.exists(path), f"Path {path} does not exist."
        return cls(path=path, **kwargs)

    def build_input(self):
        from climetlab.core.order import build_remapping

        builder = build_input(
            self.main_config.input,
            data_sources=self.main_config.get("data_sources", {}),
            order_by=self.output.order_by,
            flatten_grid=self.output.flatten_grid,
            remapping=build_remapping(self.output.remapping),
        )
        LOG.info("✅ INPUT_BUILDER")
        LOG.info(builder)
        return builder

    def build_statistics_dates(self, start, end):
        ds = open_dataset(self.path)
        dates = ds.dates

        default_start, default_end = default_statistics_dates(dates)
        if start is None:
            start = default_start
        if end is None:
            end = default_end

        start = as_first_date(start, dates)
        end = as_last_date(end, dates)

        start = start.astype(datetime.datetime)
        end = end.astype(datetime.datetime)
        return (start.isoformat(), end.isoformat())

    def read_dataset_metadata(self):
        ds = open_dataset(self.path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables
        assert len(self.variables_names) == ds.shape[1], self.dataset_shape
        self.dates = ds.dates

        z = zarr.open(self.path, "r")
        self.missing_dates = z.attrs.get("missing_dates", [])
        self.missing_dates = [np.datetime64(d) for d in self.missing_dates]

    def allow_nan(self, name):
        return name in self.main_config.statistics.get("allow_nans", [])

    @cached_property
    def registry(self):
        return ZarrBuiltRegistry(self.path)

    def initialise_dataset_backend(self):
        z = zarr.open(self.path, mode="w")
        z.create_group("_build")

    def update_metadata(self, **kwargs):
        LOG.info(f"Updating metadata {kwargs}")
        z = zarr.open(self.path, mode="w+")
        for k, v in kwargs.items():
            if isinstance(v, np.datetime64):
                v = v.astype(datetime.datetime)
            if isinstance(v, datetime.date):
                v = v.isoformat()
            z.attrs[k] = v

    def _add_dataset(self, mode="r+", **kwargs):
        z = zarr.open(self.path, mode=mode)
        return add_zarr_dataset(zarr_root=z, **kwargs)

    def get_zarr_chunks(self):
        z = zarr.open(self.path, mode="r")
        return z["data"].chunks

    def print_info(self):
        z = zarr.open(self.path, mode="r")
        try:
            LOG.info(z["data"].info)
        except Exception as e:
            LOG.info(e)


class InitialiseLoader(Loader):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.main_config = loader_config(config)

        self.statistics_registry.delete()

        LOG.info(self.main_config.dates)
        self.groups = Groups(**self.main_config.dates)

        self.output = build_output(self.main_config.output, parent=self)
        self.input = self.build_input()

        LOG.info(self.input)
        all_dates = self.groups.dates
        self.minimal_input = self.input.select([all_dates[0]])

        LOG.info(self.groups)
        LOG.info("MINIMAL INPUT :")
        LOG.info(self.minimal_input)

    def initialise(self, check_name=True):
        """Create empty dataset."""

        self.print("Config loaded ok:")
        LOG.info(self.main_config)

        dates = self.groups.dates
        frequency = dates.frequency
        assert isinstance(frequency, int), frequency

        self.print(f"Found {len(dates)} datetimes.")
        LOG.info(f"Dates: Found {len(dates)} datetimes, in {len(self.groups)} groups: ")
        LOG.info(f"Missing dates: {len(dates.missing)}")
        lengths = [len(g) for g in self.groups]
        self.print(f"Found {len(dates)} datetimes {'+'.join([str(_) for _ in lengths])}.")

        variables = self.minimal_input.variables
        self.print(f"Found {len(variables)} variables : {','.join(variables)}.")

        variables_with_nans = self.main_config.statistics.get("allow_nans", [])

        ensembles = self.minimal_input.ensembles
        self.print(f"Found {len(ensembles)} ensembles : {','.join([str(_) for _ in ensembles])}.")

        grid_points = self.minimal_input.grid_points
        LOG.info(f"gridpoints size: {[len(i) for i in grid_points]}")

        resolution = self.minimal_input.resolution
        LOG.info(f"{resolution=}")

        coords = self.minimal_input.coords
        coords["dates"] = dates
        total_shape = self.minimal_input.shape
        total_shape[0] = len(dates)
        self.print(f"total_shape = {total_shape}")

        chunks = self.output.get_chunking(coords)
        LOG.info(f"{chunks=}")
        dtype = self.output.dtype

        self.print(f"Creating Dataset '{self.path}', with {total_shape=}, {chunks=} and {dtype=}")

        metadata = {}
        metadata["uuid"] = str(uuid.uuid4())

        metadata.update(self.main_config.get("add_metadata", {}))

        metadata["_create_yaml_config"] = self.main_config.get_serialisable_dict()

        metadata["description"] = self.main_config.description
        metadata["version"] = VERSION

        metadata["data_request"] = self.minimal_input.data_request
        metadata["remapping"] = self.output.remapping

        metadata["order_by"] = self.output.order_by_as_list
        metadata["flatten_grid"] = self.output.flatten_grid

        metadata["ensemble_dimension"] = len(ensembles)
        metadata["variables"] = variables
        metadata["variables_with_nans"] = variables_with_nans
        metadata["resolution"] = resolution
        metadata["field_shape"] = self.minimal_input.field_shape
        metadata["proj_string"] = self.minimal_input.proj_string

        metadata["licence"] = self.main_config["licence"]
        metadata["attribution"] = self.main_config["attribution"]

        metadata["frequency"] = frequency
        metadata["start_date"] = dates[0].isoformat()
        metadata["end_date"] = dates[-1].isoformat()
        metadata["missing_dates"] = [_.isoformat() for _ in dates.missing]

        if check_name:
            basename, ext = os.path.splitext(os.path.basename(self.path))  # noqa: F841
            ds_name = DatasetName(
                basename,
                resolution,
                dates[0],
                dates[-1],
                frequency,
            )
            ds_name.raise_if_not_valid(print=self.print)

        if len(dates) != total_shape[0]:
            raise ValueError(
                f"Final date size {len(dates)} (from {dates[0]} to {dates[-1]}, {frequency=}) "
                f"does not match data shape {total_shape[0]}. {total_shape=}"
            )

        dates = normalize_and_check_dates(
            dates,
            metadata["start_date"],
            metadata["end_date"],
            metadata["frequency"],
        )

        metadata.update(self.main_config.get("force_metadata", {}))

        ###############################################################
        # write data
        ###############################################################

        self.initialise_dataset_backend()

        self.update_metadata(**metadata)

        self._add_dataset(name="data", chunks=chunks, dtype=dtype, shape=total_shape)
        self._add_dataset(name="dates", array=dates)
        self._add_dataset(name="latitudes", array=grid_points[0])
        self._add_dataset(name="longitudes", array=grid_points[1])

        self.registry.create(lengths=lengths)
        self.statistics_registry.create(exist_ok=False)
        self.registry.add_to_history("statistics_registry_initialised", version=self.statistics_registry.version)

        statistics_start, statistics_end = self.build_statistics_dates(
            self.main_config.statistics.get("start"),
            self.main_config.statistics.get("end"),
        )
        self.update_metadata(
            statistics_start_date=statistics_start,
            statistics_end_date=statistics_end,
        )
        LOG.info(f"Will compute statistics from {statistics_start} to {statistics_end}")

        self.registry.add_to_history("init finished")

        assert chunks == self.get_zarr_chunks(), (chunks, self.get_zarr_chunks())


class ContentLoader(Loader):
    def __init__(self, config, parts, **kwargs):
        super().__init__(**kwargs)
        self.main_config = loader_config(config)

        self.groups = Groups(**self.main_config.dates)
        self.output = build_output(self.main_config.output, parent=self)
        self.input = self.build_input()
        self.read_dataset_metadata()

        self.parts = parts
        total = len(self.registry.get_flags())
        self.cube_filter = CubesFilter(parts=self.parts, total=total)

        self.data_array = zarr.open(self.path, mode="r+")["data"]
        self.n_groups = len(self.groups)

    def load(self):
        self.registry.add_to_history("loading_data_start", parts=self.parts)

        for igroup, group in enumerate(self.groups):
            if not self.cube_filter(igroup):
                continue
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={len(self.groups)} (already done)")
                continue
            self.print(f" -> Processing {igroup} total={len(self.groups)}")
            # print("========", group)
            assert isinstance(group[0], datetime.datetime), group

            result = self.input.select(dates=group)
            assert result.dates == group, (len(result.dates), len(group))

            msg = f"Building data for group {igroup}/{self.n_groups}"
            LOG.info(msg)
            self.print(msg)

            # There are several groups.
            # There is one result to load for each group.
            self.load_result(result)
            self.registry.set_flag(igroup)

        self.registry.add_to_history("loading_data_end", parts=self.parts)
        self.registry.add_provenance(name="provenance_load")
        self.statistics_registry.add_provenance(name="provenance_load", config=self.main_config)

        self.print_info()

    def load_result(self, result):
        # There is one cube to load for each result.
        dates = result.dates

        cube = result.get_cube()
        assert cube.extended_user_shape[0] == len(dates), (
            cube.extended_user_shape[0],
            len(dates),
        )

        shape = cube.extended_user_shape
        dates_in_data = cube.user_coords["valid_datetime"]

        LOG.info(f"Loading {shape=} in {self.data_array.shape=}")

        def check_dates_in_data(lst, lst2):
            lst2 = [np.datetime64(_) for _ in lst2]
            lst = [np.datetime64(_) for _ in lst]
            assert lst == lst2, ("Dates in data are not the requested ones:", lst, lst2)

        check_dates_in_data(dates_in_data, dates)

        def dates_to_indexes(dates, all_dates):
            x = np.array(dates, dtype=np.datetime64)
            y = np.array(all_dates, dtype=np.datetime64)
            bitmap = np.isin(x, y)
            return np.where(bitmap)[0]

        indexes = dates_to_indexes(self.dates, dates_in_data)

        array = ViewCacheArray(self.data_array, shape=shape, indexes=indexes)
        self.load_cube(cube, array)

        stats = compute_statistics(array.cache, self.variables_names, allow_nan=self.allow_nan)
        self.statistics_registry.write(indexes, stats, dates=dates_in_data)

        array.flush()

    def load_cube(self, cube, array):
        # There are several cubelets for each cube
        start = time.time()
        load = 0
        save = 0

        reading_chunks = None
        total = cube.count(reading_chunks)
        self.print(f"Loading datacube: {cube}")
        bar = progress_bar(
            iterable=cube.iterate_cubelets(reading_chunks),
            total=total,
            desc=f"Loading datacube {cube}",
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
                allow_nan=self.allow_nan,
            )

            now = time.time()
            array[local_indexes] = data
            save += time.time() - now

        now = time.time()
        save += time.time() - now
        LOG.info("Written.")
        msg = f"Elapsed: {seconds(time.time() - start)}, load time: {seconds(load)}, write time: {seconds(save)}."
        self.print(msg)
        LOG.info(msg)


class StatisticsLoader(Loader):
    main_config = {}

    def __init__(
        self,
        config=None,
        statistics_output=None,
        statistics_start=None,
        statistics_end=None,
        force=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_statistics_start = statistics_start
        self.user_statistics_end = statistics_end

        self.statistics_output = statistics_output

        self.output_writer = {
            None: self.write_stats_to_dataset,
            "-": self.write_stats_to_stdout,
        }.get(self.statistics_output, self.write_stats_to_file)

        if config:
            self.main_config = loader_config(config)

        self.read_dataset_metadata()

    def _get_statistics_dates(self):
        dates = self.dates
        dtype = type(dates[0])

        def assert_dtype(d):
            assert type(d) is dtype, (type(d), dtype)

        # remove missing dates
        if self.missing_dates:
            assert type(self.missing_dates[0]) is dtype, (
                type(self.missing_dates[0]),
                dtype,
            )
        dates = [d for d in dates if d not in self.missing_dates]

        # filter dates according the the start and end dates in the metadata
        z = zarr.open(self.path, mode="r")
        start, end = z.attrs.get("statistics_start_date"), z.attrs.get("statistics_end_date")
        start, end = np.datetime64(start), np.datetime64(end)
        assert_dtype(start)
        dates = [d for d in dates if d >= start and d <= end]

        # filter dates according the the user specified start and end dates
        if self.user_statistics_start:
            limit = as_first_date(self.user_statistics_start, dates)
            limit = np.datetime64(limit)
            assert_dtype(limit)
            dates = [d for d in dates if d >= limit]

        if self.user_statistics_end:
            limit = as_last_date(self.user_statistics_start, dates)
            limit = np.datetime64(limit)
            assert_dtype(limit)
            dates = [d for d in dates if d <= limit]

        return dates

    def run(self):
        dates = self._get_statistics_dates()
        stats = self.statistics_registry.get_aggregated(dates, self.variables_names, self.allow_nan)
        self.output_writer(stats)

    def write_stats_to_file(self, stats):
        stats.save(self.statistics_output, provenance=dict(config=self.main_config))
        LOG.info(f"✅ Statistics written in {self.statistics_output}")

    def write_stats_to_dataset(self, stats):
        if self.user_statistics_start or self.user_statistics_end:
            raise ValueError(
                (
                    "Cannot write statistics in dataset with user specified dates. "
                    "This would be conflicting with the dataset metadata."
                )
            )

        if not all(self.registry.get_flags(sync=False)):
            raise Exception(f"❗Zarr {self.path} is not fully built, not writting statistics into dataset.")

        for k in ["mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            self._add_dataset(name=k, array=stats[k])

        self.registry.add_to_history("compute_statistics_end")
        LOG.info(f"Wrote statistics in {self.path}")

    def write_stats_to_stdout(self, stats):
        LOG.info(stats)


class SizeLoader(Loader):
    def __init__(self, path, print):
        self.path = path
        self.print = print

    def add_total_size(self):
        dic = compute_directory_sizes(self.path)

        size = dic["total_size"]
        n = dic["total_number_of_files"]

        LOG.info(f"Total size: {bytes(size)}")
        LOG.info(f"Total number of files: {n}")

        self.update_metadata(total_size=size, total_number_of_files=n)


class CleanupLoader(Loader):
    def run(self):
        self.statistics_registry.delete()
        self.registry.clean()
