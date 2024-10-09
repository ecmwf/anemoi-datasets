# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import json
import logging
import os
import time
import uuid
import warnings
from copy import deepcopy
from functools import cached_property

import numpy as np
import tqdm
from anemoi.utils.config import DotDict as DotDict
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


def json_tidy(o):

    if isinstance(o, datetime.datetime):
        return o.isoformat()

    if isinstance(o, datetime.datetime):
        return o.isoformat()

    if isinstance(o, datetime.timedelta):
        return frequency_to_string(o)

    raise TypeError(repr(o) + " is not JSON serializable")


def build_statistics_dates(dates, start, end):
    """Compute the start and end dates for the statistics, based on :
    - The start and end dates in the config
    - The default statistics dates convention

    Then adapt according to the actual dates in the dataset.
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


def _ignore(*args, **kwargs):
    pass


def _path_readable(path):
    import zarr

    try:
        zarr.open(path, "r")
        return True
    except zarr.errors.PathNotFoundError:
        return False


class Dataset:
    def __init__(self, path):
        self.path = path

        _, ext = os.path.splitext(self.path)
        if ext != ".zarr":
            raise ValueError(f"Unsupported extension={ext} for path={self.path}")

    def add_dataset(self, mode="r+", **kwargs):
        import zarr

        z = zarr.open(self.path, mode=mode)
        from .zarr import add_zarr_dataset

        return add_zarr_dataset(zarr_root=z, **kwargs)

    def update_metadata(self, **kwargs):
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
    def anemoi_dataset(self):
        return open_dataset(self.path)

    @cached_property
    def zarr_metadata(self):
        import zarr

        return dict(zarr.open(self.path, mode="r").attrs)

    def print_info(self):
        import zarr

        z = zarr.open(self.path, mode="r")
        try:
            LOG.info(z["data"].info)
        except Exception as e:
            LOG.info(e)

    def get_zarr_chunks(self):
        import zarr

        z = zarr.open(self.path, mode="r")
        return z["data"].chunks

    def check_name(self, resolution, dates, frequency, raise_exception=True, is_test=False):
        basename, _ = os.path.splitext(os.path.basename(self.path))
        try:
            DatasetName(basename, resolution, dates[0], dates[-1], frequency).raise_if_not_valid()
        except Exception as e:
            if raise_exception and not is_test:
                raise e
            else:
                LOG.warning(f"Dataset name error: {e}")

    def get_main_config(self):
        """Returns None if the config is not found."""
        import zarr

        z = zarr.open(self.path, mode="r")
        return loader_config(z.attrs.get("_create_yaml_config"))


class WritableDataset(Dataset):
    def __init__(self, path):
        super().__init__(path)
        self.path = path

        import zarr

        self.z = zarr.open(self.path, mode="r+")

    @cached_property
    def data_array(self):
        import zarr

        return zarr.open(self.path, mode="r+")["data"]


class NewDataset(Dataset):
    def __init__(self, path, overwrite=False):
        super().__init__(path)
        self.path = path

        import zarr

        self.z = zarr.open(self.path, mode="w")
        self.z.create_group("_build")


class Actor:  # TODO: rename to Creator
    dataset_class = WritableDataset

    def __init__(self, path, cache=None):
        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise", under="warn")

        self.path = path
        self.cache = cache
        self.dataset = self.dataset_class(self.path)

    def run(self):
        # to be implemented in the sub-classes
        raise NotImplementedError()

    def update_metadata(self, **kwargs):
        self.dataset.update_metadata(**kwargs)

    def _cache_context(self):
        from .utils import cache_context

        return cache_context(self.cache)

    def check_unkown_kwargs(self, kwargs):
        # remove this latter
        LOG.warning(f"💬 Unknown kwargs for {self.__class__.__name__}: {kwargs}")

    def read_dataset_metadata(self, path):
        ds = open_dataset(path)
        self.dataset_shape = ds.shape
        self.variables_names = ds.variables
        assert len(self.variables_names) == ds.shape[1], self.dataset_shape
        self.dates = ds.dates

        self.missing_dates = sorted(list([self.dates[i] for i in ds.missing]))

        def check_missing_dates(expected):
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
    def __init__(self, path, options=None, **kwargs):
        self.path = path
        self.options = options or {}

    def run(self):
        from .patch import apply_patch

        apply_patch(self.path, **self.options)


class Size(Actor):
    def __init__(self, path, **kwargs):
        super().__init__(path)

    def run(self):
        from .size import compute_directory_sizes

        metadata = compute_directory_sizes(self.path)
        self.update_metadata(**metadata)


class HasRegistryMixin:
    @cached_property
    def registry(self):
        from .zarr import ZarrBuiltRegistry

        return ZarrBuiltRegistry(self.path, use_threads=self.use_threads)


class HasStatisticTempMixin:
    @cached_property
    def tmp_statistics(self):
        directory = self.statistics_temp_dir or os.path.join(self.path + ".storage_for_statistics.tmp")
        return TmpStatistics(directory)


class HasElementForDataMixin:
    def create_elements(self, config):

        assert self.registry
        assert self.tmp_statistics

        LOG.info(dict(config.dates))

        self.groups = Groups(**config.dates)
        LOG.info(self.groups)

        self.output = build_output(config.output, parent=self)

        self.input = build_input_(main_config=config, output_config=self.output)
        LOG.info(self.input)


def build_input_(main_config, output_config):

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
    dataset_class = NewDataset
    def __init__(self, path, config, check_name=False, overwrite=False, use_threads=False, statistics_temp_dir=None, progress=None, test=False, cache=None, **kwargs):  # fmt: skip
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

    def run(self):
        with self._cache_context():
            return self._run()

    def _run(self):
        """Create an empty dataset of the right final shape

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
        metadata["recipe"] = sanitise(self.main_config.get_serialisable_dict())

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
    def __init__(self, path, parts=None,  use_threads=False, statistics_temp_dir=None, progress=None, cache=None, **kwargs):  # fmt: skip
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

    def run(self):
        with self._cache_context():
            self._run()

    def _run(self):
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

    def load_result(self, result):
        # There is one cube to load for each result.
        dates = list(result.group_of_dates)

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

        stats = compute_statistics(array.cache, self.variables_names, allow_nans=self._get_allow_nans())
        self.tmp_statistics.write(indexes, stats, dates=dates_in_data)

        array.flush()

    def _get_allow_nans(self):
        config = self.main_config
        if "allow_nans" in config.build:
            return config.build.allow_nans

        return config.statistics.get("allow_nans", [])

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
    def __init__(self, path, statistics_temp_dir=None, delta=[], use_threads=False, **kwargs):
        super().__init__(path)
        self.use_threads = use_threads
        self.statistics_temp_dir = statistics_temp_dir
        self.additinon_temp_dir = statistics_temp_dir
        self.actors = [
            _InitAdditions(path, delta=d, use_threads=use_threads, statistics_temp_dir=statistics_temp_dir)
            for d in delta
        ]

    def run(self):
        self.tmp_statistics.delete()
        self.registry.clean()
        for actor in self.actors:
            actor.cleanup()


class Verify(Actor):
    def __init__(self, path, **kwargs):
        super().__init__(path)

    def run(self):
        LOG.info(f"Verifying dataset at {self.path}")
        LOG.info(str(self.dataset.anemoi_dataset))


class AdditionsMixin:
    def skip(self):
        frequency = frequency_to_timedelta(self.dataset.anemoi_dataset.frequency)
        if not self.delta.total_seconds() % frequency.total_seconds() == 0:
            LOG.debug(f"Delta {self.delta} is not a multiple of frequency {frequency}. Skipping.")
            return True
        return False

    @cached_property
    def tmp_storage_path(self):
        name = "storage_for_additions"
        if self.delta:
            name += frequency_to_string(self.delta)
        return os.path.join(f"{self.path}.{name}.tmp")

    def read_from_dataset(self):
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
    def __init__(self, ds, idelta):
        self.ds = ds
        self.idelta = idelta

    def __getitem__(self, i):
        j = i - self.idelta
        if j < 0:
            raise MissingDateError(f"Missing date {j}")
        return self.ds[i : i + 1, ...] - self.ds[j : j + 1, ...]


class _InitAdditions(Actor, HasRegistryMixin, AdditionsMixin):
    def __init__(self, path, delta, use_threads=False, progress=None, **kwargs):
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress

    def run(self):
        if self.skip():
            LOG.info(f"Skipping delta={self.delta}")
            return

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=True)
        self.tmp_storage.delete()
        self.tmp_storage.create()
        LOG.info(f"Dataset {self.tmp_storage_path} additions initialized.")

    def cleanup(self):
        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        self.tmp_storage.delete()
        LOG.info(f"Cleaned temporary storage {self.tmp_storage_path}")


class _RunAdditions(Actor, HasRegistryMixin, AdditionsMixin):
    def __init__(self, path, delta, parts=None, use_threads=False, progress=None, **kwargs):
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress
        self.parts = parts

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        LOG.info(f"Writing in {self.tmp_storage_path}")

    def run(self):
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

    def allow_nans(self):
        if self.dataset.anemoi_dataset.metadata.get("allow_nans", False):
            return True

        variables_with_nans = self.dataset.anemoi_dataset.metadata.get("variables_with_nans", None)
        if variables_with_nans is not None:
            return variables_with_nans
        warnings.warn(f"❗Cannot find 'variables_with_nans' in {self.path}, assuming nans allowed.")
        return True


class _FinaliseAdditions(Actor, HasRegistryMixin, AdditionsMixin):
    def __init__(self, path, delta, use_threads=False, progress=None, **kwargs):
        super().__init__(path)
        self.delta = frequency_to_timedelta(delta)
        self.use_threads = use_threads
        self.progress = progress

        self.tmp_storage = build_storage(directory=self.tmp_storage_path, create=False)
        LOG.info(f"Reading from {self.tmp_storage_path}.")

    def run(self):
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

    def _write(self, summary):
        for k in ["mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            name = f"statistics_tendencies_{frequency_to_string(self.delta)}_{k}"
            self.dataset.add_dataset(name=name, array=summary[k], dimensions=("variable",))
        self.registry.add_to_history(f"compute_statistics_{self.__class__.__name__.lower()}_end")
        LOG.debug(f"Wrote additions in {self.path}")


def multi_addition(cls):
    class MultiAdditions:
        def __init__(self, *args, **kwargs):
            self.actors = []

            for k in kwargs.pop("delta", []):
                self.actors.append(cls(*args, delta=k, **kwargs))

            if not self.actors:
                LOG.warning("No delta found in kwargs, no additions will be computed.")

        def run(self):
            for actor in self.actors:
                actor.run()

    return MultiAdditions


InitAdditions = multi_addition(_InitAdditions)
RunAdditions = multi_addition(_RunAdditions)
FinaliseAdditions = multi_addition(_FinaliseAdditions)


class Statistics(Actor, HasStatisticTempMixin, HasRegistryMixin):
    def __init__(self, path, use_threads=False, statistics_temp_dir=None, progress=None, **kwargs):
        super().__init__(path)
        self.use_threads = use_threads
        self.progress = progress
        self.statistics_temp_dir = statistics_temp_dir

    def run(self):
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
    def allow_nans(self):
        import zarr

        z = zarr.open(self.path, mode="r")
        if "allow_nans" in z.attrs:
            return z.attrs["allow_nans"]

        if "variables_with_nans" in z.attrs:
            return z.attrs["variables_with_nans"]

        warnings.warn(f"Cannot find 'variables_with_nans' of 'allow_nans' in {self.path}.")
        return True


def chain(tasks):
    class Chain(Actor):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self):
            for cls in tasks:
                t = cls(**self.kwargs)
                t.run()

    return Chain


def creator_factory(name, trace=None, **kwargs):
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
