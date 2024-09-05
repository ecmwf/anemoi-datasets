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
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.humanize import compress_dates
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
from .statistics import fix_variance
from .utils import normalize_and_check_dates
from .writer import ViewCacheArray
from .zarr import ZarrBuiltRegistry
from .zarr import add_zarr_dataset

LOG = logging.getLogger(__name__)

VERSION = "0.20"





class GenericDatasetHandler:


    @classmethod
    def from_config(cls, *, config, path, use_threads=False, **kwargs):
        """Config is the path to the config file or a dict with the config"""

        assert isinstance(config, dict) or isinstance(config, str), config
        return cls(config=config, path=path, use_threads=use_threads, **kwargs)

    @classmethod
    def from_dataset_config(cls, *, path, use_threads=False, **kwargs):
        """Read the config saved inside the zarr dataset and instantiate the class for this config."""

        assert os.path.exists(path), f"Path {path} does not exist."
        config = read_temporary_config_from_dataset(path)
        LOG.debug("Config loaded from zarr config:\n%s", json.dumps(config, indent=4, sort_keys=True, default=str))
        return cls.from_config(config=config, path=path, use_threads=use_threads, **kwargs)

    @classmethod
    def from_dataset(cls, *, path, use_threads=False, **kwargs):
        """Instanciate the class from the path to the zarr dataset, without config."""

        assert os.path.exists(path), f"Path {path} does not exist."
        return cls(path=path, use_threads=use_threads, **kwargs)


    @cached_property
    def registry(self):
        return ZarrBuiltRegistry(self.path, use_threads=self.use_threads)

    def ready(self):
        return all(self.registry.get_flags())



    def print_info(self):
        z = zarr.open(self.path, mode="r")
        try:
            LOG.info(z["data"].info)
        except Exception as e:
            LOG.info(e)

    def _path_readable(self):
        import zarr

        try:
            zarr.open(self.path, "r")
            return True
        except zarr.errors.PathNotFoundError:
            return False


class DatasetHandler(GenericDatasetHandler):
    pass


class DatasetHandlerWithStatistics(GenericDatasetHandler):
    def __init__(self, statistics_tmp=None, **kwargs):
        super().__init__(**kwargs)
        statistics_tmp = kwargs.get("statistics_tmp") or os.path.join(self.path + ".storage_for_statistics.tmp")
        self.tmp_statistics = TmpStatistics(statistics_tmp)


class Loader(DatasetHandlerWithStatistics):

    @property
    def allow_nans(self):
        if "allow_nans" in self.main_config.build:
            return self.main_config.build.allow_nans

        return self.main_config.statistics.get("allow_nans", [])


class InitialiserLoader(Loader):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)



class ContentLoader(Loader):
    def __init__(self, config, parts, **kwargs):

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
        shape = cube.extended_user_shape
        dates_in_data = cube.user_coords["valid_datetime"]

        if cube.extended_user_shape[0] != len(dates):
            print(f"Cube shape does not match the number of dates {cube.extended_user_shape[0]}, {len(dates)}")
            print("Requested dates", compress_dates(dates))
            print("Cube dates", compress_dates(dates_in_data))

            a = set(as_datetime(_) for _ in dates)
            b = set(as_datetime(_) for _ in dates_in_data)

            print("Missing dates", compress_dates(a - b))
            print("Extra dates", compress_dates(b - a))

            raise ValueError(
                f"Cube shape does not match the number of dates {cube.extended_user_shape[0]}, {len(dates)}"
            )

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

        frequency = frequency_to_timedelta(full_ds.frequency)
        if delta is None:
            delta = frequency

        delta = frequency_to_timedelta(delta)

        if not delta.total_seconds() % frequency.total_seconds() == 0:
            raise TendenciesStatisticsDeltaNotMultipleOfFrequency(
                f"Delta {delta} is not a multiple of frequency {frequency}"
            )
        self.delta = delta
        idelta = delta.total_seconds() // frequency.total_seconds()
        assert int(idelta) == idelta, idelta
        idelta = int(idelta)

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
        delta = frequency_to_string(delta)
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
