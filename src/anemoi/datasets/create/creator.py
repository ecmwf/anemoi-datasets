# (C) Copyright 2025 Anemoi contributors.
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
import shutil
import uuid
from abc import ABC
from functools import cached_property
from typing import Any

import numpy as np
import yaml
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.sanitise import sanitise

from anemoi.datasets import MissingDateError
from anemoi.datasets import open_dataset
from anemoi.datasets.create.config import build_output
from anemoi.datasets.create.input import InputBuilder
from anemoi.datasets.dates.groups import Groups

from .gridded import DeltaDataset
from .gridded import NewDataset
from .gridded import WritableDataset
from .gridded import build_statistics_dates
from .gridded.persistent import build_storage
from .gridded.statistics import Summary
from .gridded.statistics import TmpStatistics
from .gridded.statistics import check_variance
from .gridded.statistics import compute_statistics
from .gridded.statistics import fix_variance
from .parts import PartFilter
from .utils import normalize_and_check_dates

LOG = logging.getLogger(__name__)

VERSION = "0.30"

LOG = logging.getLogger(__name__)


class Creator(ABC):
    """Abstract base class for dataset creation workflows.

    Provides methods for initialisation, loading, metadata management, statistics, and additions handling.
    """

    def __init__(self, path: str, config: dict, **kwargs: Any) -> None:
        """Initialise the Creator object.

        Parameters
        ----------
        path : str
            Path to the dataset.
        config : dict
            Main configuration dictionary.
        **kwargs : Any
            Additional keyword arguments for customisation.
        """
        # Catch all floating point errors, including overflow, sqrt(<0), etc

        np.seterr(all="raise", under="warn")

        if path.endswith("/"):
            path = path[:-1]

        self.path = path
        self.main_config = config

        # self.main_config = loader_config(config)
        self.use_threads = kwargs.pop("use_threads", False)
        self.statistics_temp_dir = kwargs.pop("statistics_temp_dir", None)
        self.addition_temp_dir = kwargs.pop("addition_temp_dir", None)
        self.parts = kwargs.pop("parts", None)

        self.kwargs = kwargs
        self.work_dir = kwargs.get("work_dir", self.path + ".work_dir")
        LOG.info(f"Using work dir: {self.work_dir}")

    #####################################################

    @classmethod
    def from_config(cls, config: dict | str, **kwargs: Any) -> "Creator":
        """Instantiate a Creator subclass from a configuration.

        Parameters
        ----------
        config : dict or str
            Configuration dictionary or path to configuration file.
        **kwargs : Any
            Additional keyword arguments for subclass initialisation.

        Returns
        -------
        Creator
            An instance of a Creator subclass.
        """
        if isinstance(config, str):
            from anemoi.datasets.create.config import loader_config

            config = loader_config(config)
            print(yaml.safe_dump(json.loads(json.dumps(config, default=str))))

        format_type = config.get("format", "gridded")
        match format_type:

            case "gridded":
                from .gridded.creator import GriddedCreator

                return GriddedCreator(config=config, **kwargs)
            case "tabular":
                from .tabular.creator import TabularCreator

                return TabularCreator(config=config, **kwargs)
            case _:
                raise ValueError(f"Unknown format type: {format_type}")

    #####################################################

    def init(self) -> int:
        """Run the initialisation process for the dataset.

        Returns
        -------
        int
            The number of groups to process.
        """
        LOG.info("Config loaded ok:")
        # LOG.info(self.main_config)

        self.dataset = self.creat_new_dataset(self.path)
        self.tmp_statistics.delete()
        """Internal method to run the initialization.

        Returns
        -------
        int
            The number of groups to process.
        """
        """Create an empty dataset of the right final shape.

        Read a small part of the data to get the shape of the data and the resolution and more metadata.
        """

        LOG.info("Cleaning temporary directories.")
        for d in [self.work_dir, self.statistics_temp_dir, self.addition_temp_dir]:
            if d is None:
                continue
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))

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

        total_shape, chunks = self.shape_and_chunks(dates)
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

    def load(self) -> None:
        """Load data into the dataset, processing each group as required."""
        self.dataset = self.open_writable_dataset(self.path)
        total = len(self.registry.get_flags())
        self.chunk_filter = PartFilter(parts=self.parts, total=total)

        self.data_array = self.dataset.data_array
        self.n_groups = len(self.groups)
        self.read_dataset_metadata(self.dataset.path)

        for igroup, group in enumerate(self.groups):
            if not self.chunk_filter(igroup):
                continue
            if self.registry.get_flag(igroup):
                LOG.info(f" -> Skipping {igroup} total={len(self.groups)} (already done)")
                continue

            # assert isinstance(group[0], datetime.datetime), type(group[0])
            LOG.debug(f"Building data for group {igroup}/{self.n_groups}")

            result = self.input.select(self.context(), argument=group)
            # BACK            assert result.group_of_dates == group, (len(result.group_of_dates), len(group), group)

            # There are several groups.
            # There is one result to load for each group.
            self.load_result(result)
            self.registry.set_flag(igroup)

        self.registry.add_provenance(name="provenance_load")
        self.tmp_statistics.add_provenance(name="provenance_load", config=self.main_config)

        self.dataset.print_info()

    def update_metadata(self, **kwargs: Any) -> None:
        """Update the metadata of the dataset.

        Parameters
        ----------
        **kwargs
            The metadata to update.
        """
        self.dataset.update_metadata(**kwargs)

    def patch(self) -> None:
        """Apply a patch to the dataset using the provided options."""
        from anemoi.datasets.create.patch import apply_patch

        options = self.kwargs.get("options", {})
        apply_patch(self.path, **options)

    def size(self) -> None:
        """Compute and update the size and constant fields metadata for the dataset."""
        """Run the size computation."""
        from anemoi.datasets.create.size import compute_directory_sizes

        self.dataset = self.open_writable_dataset(self.path)

        metadata = compute_directory_sizes(self.path)
        self.update_metadata(**metadata)

        # Look for constant fields
        ds = open_dataset(self.path)
        constants = ds.computed_constant_fields()

        variables_metadata = self.dataset.zarr_metadata.get("variables_metadata", {}).copy()
        for k in constants:
            if k in variables_metadata:
                variables_metadata[k]["constant_in_time"] = True

        self.update_metadata(constant_fields=constants, variables_metadata=variables_metadata)

    #####################################################

    @cached_property
    def groups(self) -> Groups:
        """Return the date groups for the dataset."""
        return Groups(**self.main_config.dates)

    @cached_property
    def minimal_input(self) -> Any:
        """Return a minimal input selection for a single date."""
        one_date = self.groups.one_date()
        return self.input.select(self.context(), one_date)

    @cached_property
    def output(self) -> Any:
        """Return the output builder for the dataset."""
        return build_output(self.main_config.output, parent=self)

    @cached_property
    def input(self) -> InputBuilder:
        """Return the input builder for the dataset."""

        return InputBuilder(
            self.main_config.input,
            data_sources=self.main_config.get("data_sources", {}),
        )

    def cleanup(self) -> None:
        """Clean up temporary statistics and registry, and remove additions if specified."""
        self.tmp_statistics.delete()
        self.registry.clean()
        for delta in self.kwargs.get("delta", []):
            self._cleanup_addition(frequency_to_timedelta(delta))

        for d in [self.work_dir, self.statistics_temp_dir, self.addition_temp_dir]:
            if d is None:
                continue
            if os.path.exists(d):
                shutil.rmtree(d)

    def verify(self) -> None:
        """Run verification on the dataset and log the results."""
        """Run the verification."""
        self.dataset = self.open_writable_dataset(self.path)
        LOG.info(f"Verifying dataset at {self.path}")
        LOG.info(str(self.dataset.anemoi_dataset))

    def skip(self, delta: datetime.timedelta) -> bool:
        """Check if the additions should be skipped.

        Parameters
        ----------
        delta : datetime.timedelta
            The delta to check.

        Returns
        -------
        bool
            Whether to skip the additions.
        """
        frequency = frequency_to_timedelta(self.dataset.anemoi_dataset.frequency)
        if not delta.total_seconds() % frequency.total_seconds() == 0:
            LOG.debug(f"Delta {delta} is not a multiple of frequency {frequency}. Skipping.")
            return True

        if self.dataset.zarr_metadata.get("build", {}).get("additions", None) is False:
            LOG.warning(f"Additions are disabled for {self.path} in the recipe.")
            return True

        return False

    def tmp_storage_path(self, delta: datetime.timedelta | None = None) -> str:
        """Get the path to the temporary storage for additions.

        Parameters
        ----------
        delta : datetime.timedelta, optional
            The delta for which to get the storage path.

        Returns
        -------
        str
            The path to the temporary storage.
        """
        """Get the path to the temporary storage."""
        name = "storage_for_additions"
        if delta:
            name += frequency_to_string(delta)
        return os.path.join(f"{self.path}.{name}.tmp")

    def read_from_dataset(self, delta: datetime.timedelta) -> None:
        """Read data from the dataset for a given delta.

        Parameters
        ----------
        delta : datetime.timedelta
            The delta to read from the dataset.
        """
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

        idelta = delta.total_seconds() // self.frequency.total_seconds()
        assert int(idelta) == idelta, idelta
        idelta = int(idelta)
        self.ds = DeltaDataset(ds, idelta)

    def init_additions(self) -> None:
        """Initialise temporary storage and prepare for additions for all specified deltas."""
        build_storage(directory=self.tmp_storage_path(), create=True)
        self.dataset = self.open_writable_dataset(self.path)
        for delta in self.kwargs.get("delta", []):
            self._init_addition(frequency_to_timedelta(delta))

    def _init_addition(self, delta: datetime.timedelta) -> None:
        """Run the additions initialisation for a specific delta.

        Parameters
        ----------
        delta : datetime.timedelta
            The delta for which to initialise additions.
        """
        """Run the additions initialization."""
        if self.skip(delta):
            LOG.info(f"Skipping {delta=}")
            return

        self.tmp_storage = build_storage(directory=self.tmp_storage_path(delta), create=True)
        self.tmp_storage.delete()
        self.tmp_storage.create()
        LOG.info(f"Dataset {self.tmp_storage_path(delta)} additions initialised.")

    def _cleanup_addition(self, delta: datetime.timedelta) -> None:
        """Clean up the temporary storage for a specific delta.

        Parameters
        ----------
        delta : datetime.timedelta
            The delta for which to clean up storage.
        """
        """Clean up the temporary storage."""
        self.tmp_storage = build_storage(directory=self.tmp_storage_path(delta), create=False)
        self.tmp_storage.delete()
        LOG.info(f"Cleaned temporary storage {self.tmp_storage_path(delta)}")

    def load_additions(self) -> None:
        """Load additions for all specified deltas into the dataset."""

        self.dataset = self.open_writable_dataset(self.path)
        for delta in self.kwargs.get("delta", []):
            self._load_addition(frequency_to_timedelta(delta))

    def _load_addition(self, delta: datetime.timedelta) -> None:
        """Run the additions process for a specific delta.

        Parameters
        ----------
        delta : datetime.timedelta
            The delta for which to load additions.
        """
        """Run the additions."""
        if self.skip(delta):
            LOG.info(f"Skipping {delta=}")
            return

        self.read_from_dataset(delta)
        self.tmp_storage = build_storage(directory=self.tmp_storage_path(delta), create=False)
        LOG.info(f"Writing in {self.tmp_storage_path(delta)}")
        chunk_filter = PartFilter(parts=self.parts, total=self.total)
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

    def finalise_additions(self) -> None:
        """Finalise additions for all specified deltas, aggregating and writing statistics."""
        self.dataset = self.open_writable_dataset(self.path)
        for delta in self.kwargs.get("delta", []):
            self._finalise_addition(frequency_to_timedelta(delta))

    def _finalise_addition(self, delta: datetime.timedelta) -> None:
        """Run the additions finalisation for a specific delta, aggregating statistics and writing results.

        Parameters
        ----------
        delta : datetime.timedelta
            The delta for which to finalise additions.
        """
        """Run the additions finalization."""
        if self.skip(delta):
            LOG.info(f"Skipping {delta=}.")
            return

        self.read_from_dataset(delta=delta)
        self.tmp_storage = build_storage(directory=self.tmp_storage_path(delta), create=False)

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
        self._write(self.summary, delta)
        self.tmp_storage.delete()

    def _write(self, summary: Summary, delta: datetime.timedelta) -> None:
        """Write the summary statistics to the dataset for a given delta.

        Parameters
        ----------
        summary : Summary
            The summary to write.
        delta : datetime.timedelta
            The delta value.
        """
        """Write the summary to the dataset.

        Parameters
        ----------
        summary : Summary
            The summary to write.
        delta : datetime.timedelta
            The delta value.
        """
        for k in ["mean", "stdev", "minimum", "maximum", "sums", "squares", "count", "has_nans"]:
            name = f"statistics_tendencies_{frequency_to_string(delta)}_{k}"
            self.dataset.add_dataset(name=name, array=summary[k], dimensions=("variable",))
        self.registry.add_to_history(f"compute_statistics_{self.__class__.__name__.lower()}_end")
        LOG.debug(f"Wrote additions in {self.path}")

    def statistics(self) -> None:
        """Run the statistics computation and write results to the dataset."""
        self.dataset = self.open_writable_dataset(self.path)
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
            self.dataset.add_dataset(name=k, array=stats[k], dimensions=("variable",))

        self.registry.add_to_history("compute_statistics_end")
        LOG.info(f"Wrote statistics in {self.path}")

    def creat_new_dataset(self, path: str) -> NewDataset:
        """Create a new dataset at the specified path.

        Parameters
        ----------
        path : str
            Path to the new dataset.

        Returns
        -------
        NewDataset
            The created NewDataset instance.
        """
        return NewDataset(path)

    def open_writable_dataset(self, path: str) -> WritableDataset:
        """Open a writable dataset at the specified path.

        Parameters
        ----------
        path : str
            Path to the writable dataset.

        Returns
        -------
        WritableDataset
            The opened WritableDataset instance.
        """
        return WritableDataset(path)

    @cached_property
    def registry(self) -> Any:
        """Get the registry for the dataset."""
        """Get the registry."""
        from .gridded.zarr import ZarrBuiltRegistry

        return ZarrBuiltRegistry(self.path, use_threads=self.use_threads)

    @cached_property
    def tmp_statistics(self) -> TmpStatistics:
        """Get the temporary statistics object for the dataset."""
        """Get the temporary statistics."""
        directory = self.statistics_temp_dir or os.path.join(self.path + ".storage_for_statistics.tmp")
        return TmpStatistics(directory)

    def read_dataset_metadata(self, path: str) -> None:
        """Read the metadata of the dataset and check for missing dates consistency.

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
