# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
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

import numpy as np
import tqdm
import zarr
from anemoi.utils.config import DotDict
from anemoi.utils.humanize import seconds_to_human

from anemoi.datasets import MissingDateError
from anemoi.datasets import open_dataset
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
from .utils import normalize_and_check_dates
from .writer import ViewCacheArray
from .zarr import ZarrBuiltRegistry
from .zarr import add_zarr_dataset

LOG = logging.getLogger(__name__)

VERSION = "0.20"


def set_to_test_mode(cfg):
    NUMBER_OF_DATES = 4

    dates = cfg.dates
    LOG.warn(f"Running in test mode. Changing the list of dates to use only {NUMBER_OF_DATES}.")
    groups = Groups(**cfg.dates)
    dates = groups.dates
    cfg.dates = dict(
        start=dates[0],
        end=dates[NUMBER_OF_DATES - 1],
        frequency=dates.frequency,
        group_by=NUMBER_OF_DATES,
    )

    def set_element_to_test(obj):
        if isinstance(obj, (list, tuple)):
            for v in obj:
                set_element_to_test(v)
            return
        if isinstance(obj, (dict, DotDict)):
            if "grid" in obj:
                previous = obj["grid"]
                obj["grid"] = "20./20."
                LOG.warn(f"Running in test mode. Setting grid to {obj['grid']} instead of {previous}")
            if "number" in obj:
                if isinstance(obj["number"], (list, tuple)):
                    previous = obj["number"]
                    obj["number"] = previous[0:3]
                    LOG.warn(f"Running in test mode. Setting number to {obj['number']} instead of {previous}")
            for k, v in obj.items():
                set_element_to_test(v)
            if "constants" in obj:
                constants = obj["constants"]
                if "param" in constants and isinstance(constants["param"], list):
                    constants["param"] = ["cos_latitude"]

    set_element_to_test(cfg)


class GenericDatasetHandler:
    def __init__(self, *, path, use_threads=False, **kwargs):

        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise", under="warn")

        assert isinstance(path, str), path

        self.path = path
        self.kwargs = kwargs
        self.use_threads = use_threads
        if "test" in kwargs:
            self.test = kwargs["test"]

    @classmethod
    def from_config(cls, *, config, path, use_threads=False, **kwargs):
        """Config is the path to the config file or a dict with the config"""

        assert isinstance(config, dict) or isinstance(config, str), config
        return cls(config=config, path=path, use_threads=use_threads, **kwargs)

    @classmethod
    def from_dataset_config(cls, *, path, use_threads=False, **kwargs):
        """Read the config saved inside the zarr dataset and instantiate the class for this config."""

        assert os.path.exists(path), f"Path {path} does not exist."
        z = zarr.open(path, mode="r")
        config = z.attrs["_create_yaml_config"]
        LOG.debug("Config loaded from zarr config:\n%s", json.dumps(config, indent=4, sort_keys=True, default=str))
        return cls.from_config(config=config, path=path, use_threads=use_threads, **kwargs)

    @classmethod
    def from_dataset(cls, *, path, use_threads=False, **kwargs):
        """Instanciate the class from the path to the zarr dataset, without config."""

        assert os.path.exists(path), f"Path {path} does not exist."
        return cls(path=path, use_threads=use_threads, **kwargs)

    def read_dataset_metadata(self):
        ds = open_dataset(self.path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables
        assert len(self.variables_names) == ds.shape[1], self.dataset_shape
        self.dates = ds.dates

        self.missing_dates = sorted(list([self.dates[i] for i in ds.missing]))

        z = zarr.open(self.path, "r")
        missing_dates = z.attrs.get("missing_dates", [])
        missing_dates = sorted([np.datetime64(d) for d in missing_dates])

        if missing_dates != self.missing_dates:
            LOG.warn("Missing dates given in recipe do not match the actual missing dates in the dataset.")
            LOG.warn(f"Missing dates in recipe: {sorted(str(x) for x in missing_dates)}")
            LOG.warn(f"Missing dates in dataset: {sorted(str(x) for x in  self.missing_dates)}")
            raise ValueError("Missing dates given in recipe do not match the actual missing dates in the dataset.")

    @cached_property
    def registry(self):
        return ZarrBuiltRegistry(self.path, use_threads=self.use_threads)

    def ready(self):
        return all(self.registry.get_flags())

    def update_metadata(self, **kwargs):
        LOG.debug(f"Updating metadata {kwargs}")
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


class DatasetHandler(GenericDatasetHandler):
    pass


class DatasetHandlerWithStatistics(GenericDatasetHandler):
    def __init__(self, statistics_tmp=None, **kwargs):
        super().__init__(**kwargs)
        statistics_tmp = kwargs.get("statistics_tmp") or os.path.join(self.path + ".storage_for_statistics.tmp")
        self.tmp_statistics = TmpStatistics(statistics_tmp)


class Loader(DatasetHandlerWithStatistics):
    def build_input(self):
        from earthkit.data.core.order import build_remapping

        builder = build_input(
            self.main_config.input,
            data_sources=self.main_config.get("data_sources", {}),
            order_by=self.output.order_by,
            flatten_grid=self.output.flatten_grid,
            remapping=build_remapping(self.output.remapping),
            use_grib_paramid=self.main_config.build.use_grib_paramid,
        )
        LOG.debug("✅ INPUT_BUILDER")
        LOG.debug(builder)
        return builder

    @property
    def allow_nans(self):
        if "allow_nans" in self.main_config.build:
            return self.main_config.build.allow_nans

        return self.main_config.statistics.get("allow_nans", [])


class InitialiserLoader(Loader):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.main_config = loader_config(config)
        if self.test:
            set_to_test_mode(self.main_config)

        LOG.info(dict(self.main_config.dates))

        self.tmp_statistics.delete()

        self.groups = Groups(**self.main_config.dates)
        LOG.info(self.groups)

        self.output = build_output(self.main_config.output, parent=self)
        self.input = self.build_input()
        LOG.info(self.input)

        first_date = self.groups.dates[0]
        self.minimal_input = self.input.select([first_date])
        LOG.info("Minimal input (using only the first date) :")
        LOG.info(self.minimal_input)

    def build_statistics_dates(self, start, end):
        """Compute the start and end dates for the statistics, based on :
        - The start and end dates in the config
        - The default statistics dates convention

        Then adapt according to the actual dates in the dataset.
        """

        ds = open_dataset(self.path)
        dates = ds.dates

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

    def initialise_dataset_backend(self):
        z = zarr.open(self.path, mode="w")
        z.create_group("_build")

    def initialise(self, check_name=True):
        """Create an empty dataset of the right final shape

        Read a small part of the data to get the shape of the data and the resolution and more metadata.
        """

        LOG.info("Config loaded ok:")
        # LOG.info(self.main_config)

        dates = self.groups.dates
        frequency = dates.frequency
        assert isinstance(frequency, int), frequency

        LOG.info(f"Found {len(dates)} datetimes.")
        LOG.info(f"Dates: Found {len(dates)} datetimes, in {len(self.groups)} groups: ")
        LOG.info(f"Missing dates: {len(dates.missing)}")
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

        metadata["start_date"] = dates[0].isoformat()
        metadata["end_date"] = dates[-1].isoformat()
        metadata["frequency"] = frequency
        metadata["missing_dates"] = [_.isoformat() for _ in dates.missing]

        metadata["version"] = VERSION

        if check_name:
            basename, ext = os.path.splitext(os.path.basename(self.path))  # noqa: F841
            ds_name = DatasetName(basename, resolution, dates[0], dates[-1], frequency)
            ds_name.raise_if_not_valid()

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

        self.initialise_dataset_backend()

        self.update_metadata(**metadata)

        self._add_dataset(
            name="data",
            chunks=chunks,
            dtype=dtype,
            shape=total_shape,
            dimensions=("time", "variable", "ensemble", "cell"),
        )
        self._add_dataset(name="dates", array=dates, dimensions=("time",))
        self._add_dataset(name="latitudes", array=grid_points[0], dimensions=("cell",))
        self._add_dataset(name="longitudes", array=grid_points[1], dimensions=("cell",))

        self.registry.create(lengths=lengths)
        self.tmp_statistics.create(exist_ok=False)
        self.registry.add_to_history("tmp_statistics_initialised", version=self.tmp_statistics.version)

        statistics_start, statistics_end = self.build_statistics_dates(
            self.main_config.statistics.get("start"),
            self.main_config.statistics.get("end"),
        )
        self.update_metadata(statistics_start_date=statistics_start, statistics_end_date=statistics_end)
        LOG.info(f"Will compute statistics from {statistics_start} to {statistics_end}")

        self.registry.add_to_history("init finished")

        assert chunks == self.get_zarr_chunks(), (chunks, self.get_zarr_chunks())

        # Return the number of groups to process, so we can show a nice progress bar
        return len(lengths)


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
        self.chunk_filter = ChunkFilter(parts=self.parts, total=total)

        self.data_array = zarr.open(self.path, mode="r+")["data"]
        self.n_groups = len(self.groups)

    def load(self):
        for igroup, group in enumerate(self.groups):
            if not self.chunk_filter(igroup):
                continue
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={len(self.groups)} (already done)")
                continue

            assert isinstance(group[0], datetime.datetime), group

            result = self.input.select(dates=group)
            assert result.dates == group, (len(result.dates), len(group))

            LOG.debug(f"Building data for group {igroup}/{self.n_groups}")

            # There are several groups.
            # There is one result to load for each group.
            self.load_result(result)
            self.registry.set_flag(igroup)

        self.registry.add_provenance(name="provenance_load")
        self.tmp_statistics.add_provenance(name="provenance_load", config=self.main_config)

        # self.print_info()

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

        LOG.debug(f"Loading {shape=} in {self.data_array.shape=}")

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

        stats = compute_statistics(array.cache, self.variables_names, allow_nans=self.allow_nans)
        self.tmp_statistics.write(indexes, stats, dates=dates_in_data)

        array.flush()

    def load_cube(self, cube, array):
        # There are several cubelets for each cube
        start = time.time()
        load = 0
        save = 0

        reading_chunks = None
        total = cube.count(reading_chunks)
        LOG.debug(f"Loading datacube: {cube}")

        def position(x):
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
                allow_nans=self.allow_nans,
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


class StatisticsAdder(DatasetHandlerWithStatistics):
    def __init__(
        self,
        statistics_output=None,
        statistics_start=None,
        statistics_end=None,
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

        self.read_dataset_metadata()

    @cached_property
    def allow_nans(self):
        z = zarr.open(self.path, mode="r")
        if "allow_nans" in z.attrs:
            return z.attrs["allow_nans"]

        if "variables_with_nans" in z.attrs:
            return z.attrs["variables_with_nans"]

        warnings.warn(f"Cannot find 'variables_with_nans' of 'allow_nans' in {self.path}.")
        return True

    def _get_statistics_dates(self):
        dates = self.dates
        dtype = type(dates[0])

        def assert_dtype(d):
            assert type(d) is dtype, (type(d), dtype)

        # remove missing dates
        if self.missing_dates:
            assert_dtype(self.missing_dates[0])
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
        stats = self.tmp_statistics.get_aggregated(dates, self.variables_names, self.allow_nans)
        self.output_writer(stats)

    def write_stats_to_file(self, stats):
        stats.save(self.statistics_output)
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

        for k in [
            "mean",
            "stdev",
            "minimum",
            "maximum",
            "sums",
            "squares",
            "count",
            "has_nans",
        ]:
            self._add_dataset(name=k, array=stats[k], dimensions=("variable",))

        self.registry.add_to_history("compute_statistics_end")
        LOG.info(f"Wrote statistics in {self.path}")

    def write_stats_to_stdout(self, stats):
        LOG.info(stats)


class GenericAdditions(GenericDatasetHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=True)

    @property
    def tmp_storage_path(self):
        """This should be implemented in the subclass."""
        raise NotImplementedError()

    @property
    def final_storage_path(self):
        """This should be implemented in the subclass."""
        raise NotImplementedError()

    def initialise(self):
        self.tmp_storage.delete()
        self.tmp_storage.create()
        LOG.info(f"Dataset {self.path} additions initialized.")

    def run(self, parts):
        """This should be implemented in the subclass."""
        raise NotImplementedError()

    def finalise(self):

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
            LOG.warn(f"Not enough data found in {self.path} to compute {self.__class__.__name__}. Skipped.")
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
        # remove negative variance due to numerical errors
        # x[- 1e-15 < (x / (np.sqrt(squares / count) + np.abs(mean))) < 0] = 0
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
        self.check_statistics()
        self._write(self.summary)
        self.tmp_storage.delete()

    def _write(self, summary):
        for k in [
            "mean",
            "stdev",
            "minimum",
            "maximum",
            "sums",
            "squares",
            "count",
            "has_nans",
        ]:
            name = self.final_storage_name(k)
            self._add_dataset(name=name, array=summary[k], dimensions=("variable",))
        self.registry.add_to_history(f"compute_statistics_{self.__class__.__name__.lower()}_end")
        LOG.debug(f"Wrote additions in {self.path} ({self.final_storage_name('*')})")

    def check_statistics(self):
        pass

    @cached_property
    def _variables_with_nans(self):
        z = zarr.open(self.path, mode="r")
        if "variables_with_nans" in z.attrs:
            return z.attrs["variables_with_nans"]
        return None

    @cached_property
    def _allow_nans(self):
        z = zarr.open(self.path, mode="r")
        return z.attrs.get("allow_nans", False)

    def allow_nans(self):

        if self._allow_nans:
            return True

        if self._variables_with_nans is not None:
            return self._variables_with_nans
        warnings.warn(f"❗Cannot find 'variables_with_nans' in {self.path}, assuming nans allowed.")
        return True


class StatisticsAddition(GenericAdditions):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        z = zarr.open(self.path, mode="r")
        start = z.attrs["statistics_start_date"]
        end = z.attrs["statistics_end_date"]
        self.ds = open_dataset(self.path, start=start, end=end)

        self.variables = self.ds.variables
        self.dates = self.ds.dates

        assert len(self.variables) == self.ds.shape[1], self.ds.shape
        self.total = len(self.dates)

    @property
    def tmp_storage_path(self):
        return f"{self.path}.storage_statistics.tmp"

    def final_storage_name(self, k):
        return k

    def run(self, parts):
        chunk_filter = ChunkFilter(parts=parts, total=self.total)
        for i in range(0, self.total):
            if not chunk_filter(i):
                continue
            date = self.dates[i]
            try:
                arr = self.ds[i : i + 1, ...]
                stats = compute_statistics(arr, self.variables, allow_nans=self.allow_nans)
                self.tmp_storage.add([date, i, stats], key=date)
            except MissingDateError:
                self.tmp_storage.add([date, i, "missing"], key=date)
        self.tmp_storage.flush()
        LOG.debug(f"Dataset {self.path} additions run.")

    def check_statistics(self):
        ds = open_dataset(self.path)
        ref = ds.statistics
        for k in ds.statistics:
            assert np.all(np.isclose(ref[k], self.summary[k], rtol=1e-4, atol=1e-4)), (
                k,
                ref[k],
                self.summary[k],
            )


class DeltaDataset:
    def __init__(self, ds, idelta):
        self.ds = ds
        self.idelta = idelta

    def __getitem__(self, i):
        j = i - self.idelta
        if j < 0:
            raise MissingDateError(f"Missing date {j}")
        return self.ds[i : i + 1, ...] - self.ds[j : j + 1, ...]


class TendenciesStatisticsDeltaNotMultipleOfFrequency(ValueError):
    pass


class TendenciesStatisticsAddition(GenericAdditions):
    def __init__(self, path, delta=None, **kwargs):
        full_ds = open_dataset(path)
        self.variables = full_ds.variables

        frequency = full_ds.frequency
        if delta is None:
            delta = frequency
        assert isinstance(delta, int), delta
        if not delta % frequency == 0:
            raise TendenciesStatisticsDeltaNotMultipleOfFrequency(
                f"Delta {delta} is not a multiple of frequency {frequency}"
            )
        self.delta = delta
        idelta = delta // frequency

        super().__init__(path=path, **kwargs)

        z = zarr.open(self.path, mode="r")
        start = z.attrs["statistics_start_date"]
        end = z.attrs["statistics_end_date"]
        start = datetime.datetime.fromisoformat(start)
        ds = open_dataset(self.path, start=start, end=end)
        self.dates = ds.dates
        self.total = len(self.dates)

        ds = open_dataset(self.path, start=start, end=end)
        self.ds = DeltaDataset(ds, idelta)

    @property
    def tmp_storage_path(self):
        return f"{self.path}.storage_statistics_{self.delta}h.tmp"

    def final_storage_name(self, k):
        return self.final_storage_name_from_delta(k, delta=self.delta)

    @classmethod
    def final_storage_name_from_delta(_, k, delta):
        if isinstance(delta, int):
            delta = str(delta)
        if not delta.endswith("h"):
            delta = delta + "h"
        return f"statistics_tendencies_{delta}_{k}"

    def run(self, parts):
        chunk_filter = ChunkFilter(parts=parts, total=self.total)
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


class DatasetVerifier(GenericDatasetHandler):

    def verify(self):
        pass
