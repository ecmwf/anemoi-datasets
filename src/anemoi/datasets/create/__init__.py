# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
VERSION = "0.20"

import logging
from copy import deepcopy
from functools import cached_property
import numpy as np
from anemoi.datasets.dates.groups import Groups
import os
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

LOG = logging.getLogger(__name__)


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


def _ignore(*args, **kwargs):
    pass


def creator_factory(name, **kwargs):
    cls = dict(
        init=Init,
        load=Load,
    )[name]
    return cls(**kwargs)


def _path_readable(path):
    import zarr

    try:
        zarr.open(path, "r")
        return True
    except zarr.errors.PathNotFoundError:
        return False


class Actor:
    cache = None

    def __init__(self, path):
        # Catch all floating point errors, including overflow, sqrt(<0), etc
        np.seterr(all="raise", under="warn")

        self.path = path

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

        z = zarr.open(path, "r")
        missing_dates = z.attrs.get("missing_dates", [])
        missing_dates = sorted([np.datetime64(d) for d in missing_dates])

        if missing_dates != self.missing_dates:
            LOG.warn("Missing dates given in recipe do not match the actual missing dates in the dataset.")
            LOG.warn(f"Missing dates in recipe: {sorted(str(x) for x in missing_dates)}")
            LOG.warn(f"Missing dates in dataset: {sorted(str(x) for x in  self.missing_dates)}")
            raise ValueError("Missing dates given in recipe do not match the actual missing dates in the dataset.")


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
                LOG.error(f"Error in dataset name: {e}")

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


def build_input_(main_config, output_config):
    from earthkit.data.core.order import build_remapping

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
    def __init__(self, path, config, check_name=False, overwrite=False, use_threads=False, statistics_temp_dir=None, progress=None, test=False, **kwargs):  # fmt: skip
        super().__init__(path)
        self.config = config
        self.check_name = check_name
        self.use_threads = use_threads
        self.statistics_temp_dir = statistics_temp_dir
        self.progress = progress
        self.test = test
        self.check_unkown_kwargs(kwargs)

        if _path_readable(path) and not overwrite:
            raise Exception(f"{self.path} already exists. Use overwrite=True to overwrite.")

        self.dataset = NewDataset(self.path)

        self.main_config = loader_config(config)
        if self.test:
            set_to_test_mode(self.main_config)

        # self.registry.delete() ??
        self.tmp_statistics.delete()

        assert isinstance(self.main_config.output.order_by, dict), self.main_config.output.order_by
        self.create_elements(self.main_config)

        first_date = self.groups.dates[0]
        self.minimal_input = self.input.select([first_date])
        LOG.info("Minimal input for 'init' step (using only the first date) :")
        LOG.info(self.minimal_input)

    def run_it(self):
        with self._cache_context():
            self._run()

    def _run(self):
        """Create an empty dataset of the right final shape

        Read a small part of the data to get the shape of the data and the resolution and more metadata.
        """

        LOG.info("Config loaded ok:")
        # LOG.info(self.main_config)

        dates = self.groups.dates
        frequency = dates.frequency
        assert isinstance(frequency, datetime.timedelta), frequency

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

        def check_config(a,b):
            a = json.dumps(a, sort_keys=True, default=str)
            b = json.dumps(b , sort_keys=True, default=str)
            if a != b:
                print("❌❌❌ FIXME: DIFFERENT CONFIGS (group_by different because of test)")
                print(a)
                print(b)
        check_config(self.main_config,self.dataset.get_main_config())

        # Return the number of groups to process, so we can show a nice progress bar
        return len(lengths)


class Load(Actor, HasRegistryMixin, HasStatisticTempMixin, HasElementForDataMixin):
    def __init__(self, path, parts=None,  use_threads=False, statistics_temp_dir=None, progress=None, **kwargs):  # fmt: skip
        super().__init__(path)
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


class Creator:
    def __init__( self, path, config=None, cache=None, use_threads=False, statistics_tmp=None, overwrite=False, test=None, progress=None, **kwargs):  # fmt: skip
        self.path = path  # Output path
        self.config = config
        self.cache = cache
        self.use_threads = use_threads
        self.statistics_tmp = statistics_tmp
        self.overwrite = overwrite
        self.test = test
        self.progress = progress if progress is not None else _ignore
        self.kwargs = kwargs

    def check_unkown_kwargs(self):
        for k in self.kwargs:
            raise Exception(f"Unknown kwargs: {self.kwargs}")

    def init(self):
        check_name = self.kwargs.pop("check_name", False)
        self.check_unkown_kwargs()

        from .loaders import InitialiserLoader

        with self._cache_context():
            obj = InitialiserLoader.from_config(
                path=self.path,
                config=self.config,
                statistics_tmp=self.statistics_tmp,
                use_threads=self.use_threads,
                progress=self.progress,
                test=self.test,
            )
            return obj.initialise(check_name=check_name)

    def load(self):
        parts = self.kwargs.pop("parts", None)
        from .loaders import ContentLoader

        with self._cache_context():
            loader = ContentLoader.from_dataset_config(
                path=self.path,
                statistics_tmp=self.statistics_tmp,
                use_threads=self.use_threads,
                progress=self.progress,
                parts=parts,
            )
            loader.load()

    def statistics(self):
        output = self.kwargs.pop("output", None)
        start = self.kwargs.pop("start", None)
        end = self.kwargs.pop("end", None)
        self.check_unkown_kwargs()

        from .loaders import StatisticsAdder

        loader = StatisticsAdder.from_dataset(
            path=self.path,
            use_threads=self.use_threads,
            progress=self.progress,
            statistics_tmp=self.statistics_tmp,
            statistics_output=output,
            recompute=False,
            statistics_start=start,
            statistics_end=end,
        )
        loader.run()
        assert loader.ready()

    def size(self):
        from .loaders import DatasetHandler
        from .size import compute_directory_sizes

        metadata = compute_directory_sizes(self.path)
        handle = DatasetHandler.from_dataset(path=self.path, use_threads=self.use_threads)
        handle.update_metadata(**metadata)

    def cleanup(self):
        from .loaders import DatasetHandlerWithStatistics

        cleaner = DatasetHandlerWithStatistics.from_dataset(
            path=self.path,
            use_threads=self.use_threads,
            progress=self.progress,
            statistics_tmp=self.statistics_tmp,
        )
        cleaner.tmp_statistics.delete()
        cleaner.registry.clean()

    def patch(self):
        from .patch import apply_patch

        apply_patch(self.path, **self.kwargs)

    def init_additions(self):
        delta = self.kwargs.pop("delta", [])
        recompute_statistics = self.kwargs.pop("recompute_statistics", False)
        self.check_unkown_kwargs()

        from .loaders import StatisticsAddition
        from .loaders import TendenciesStatisticsAddition
        from .loaders import TendenciesStatisticsDeltaNotMultipleOfFrequency

        if recompute_statistics:
            a = StatisticsAddition.from_dataset(path=self.path, use_threads=self.use_threads)
            a.initialise()

        for d in delta:
            try:
                a = TendenciesStatisticsAddition.from_dataset(
                    path=self.path,
                    use_threads=self.use_threads,
                    progress=self.progress,
                    delta=d,
                )
                a.initialise()
            except TendenciesStatisticsDeltaNotMultipleOfFrequency:
                LOG.info(f"Skipping delta={d} as it is not a multiple of the frequency.")

    def run_additions(self):
        delta = self.kwargs.pop("delta", [])
        recompute_statistics = self.kwargs.pop("recompute_statistics", False)
        parts = self.kwargs.pop("parts", None)
        self.check_unkown_kwargs()

        from .loaders import StatisticsAddition
        from .loaders import TendenciesStatisticsAddition
        from .loaders import TendenciesStatisticsDeltaNotMultipleOfFrequency

        if recompute_statistics:
            a = StatisticsAddition.from_dataset(path=self.path, use_threads=self.use_threads)
            a.run(parts)

        for d in delta:
            try:
                a = TendenciesStatisticsAddition.from_dataset(
                    path=self.path,
                    use_threads=self.use_threads,
                    progress=self.progress,
                    delta=d,
                )
                a.run(parts)
            except TendenciesStatisticsDeltaNotMultipleOfFrequency:
                LOG.debug(f"Skipping delta={d} as it is not a multiple of the frequency.")

    def finalise_additions(self):
        delta = self.kwargs.pop("delta", [])
        recompute_statistics = self.kwargs.pop("recompute_statistics", False)
        self.check_unkown_kwargs()

        from .loaders import StatisticsAddition
        from .loaders import TendenciesStatisticsAddition
        from .loaders import TendenciesStatisticsDeltaNotMultipleOfFrequency

        if recompute_statistics:
            a = StatisticsAddition.from_dataset(path=self.path, use_threads=self.use_threads)
            a.finalise()

        for d in delta:
            try:
                a = TendenciesStatisticsAddition.from_dataset(
                    path=self.path,
                    use_threads=self.use_threads,
                    progress=self.progress,
                    delta=d,
                )
                a.finalise()
            except TendenciesStatisticsDeltaNotMultipleOfFrequency:
                LOG.debug(f"Skipping delta={d} as it is not a multiple of the frequency.")

    def finalise(self):
        self.statistics()
        self.size()

    def create(self):
        self.init()
        self.load()
        self.finalise()
        self.additions()
        self.cleanup()

    def additions(self):
        # from .loaders import read_temporary_config_from_dataset
        # config = read_temporary_config_from_dataset(self.path)

        self.init_additions()
        self.run_additions()
        self.finalise_additions()

    def _cache_context(self):
        from .utils import cache_context

        return cache_context(self.cache)

    def verify(self):
        from .loaders import DatasetVerifier

        handle = DatasetVerifier.from_dataset(path=self.path, use_threads=self.use_threads)

        handle.verify()
