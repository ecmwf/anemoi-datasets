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
from functools import cached_property
from typing import Any

import cftime
import numpy as np
import zarr
from anemoi.utils.dates import frequency_to_string
from earthkit.data.core.order import build_remapping

from anemoi.datasets import open_dataset
from anemoi.datasets.create.base import Task
from anemoi.datasets.create.check import DatasetName
from anemoi.datasets.create.config import build_output
from anemoi.datasets.create.config import loader_config
from anemoi.datasets.create.gridded.context import FieldContext
from anemoi.datasets.create.input import InputBuilder
from anemoi.datasets.create.statistics import TmpStatistics
from anemoi.datasets.create.statistics import default_statistics_dates
from anemoi.datasets.dates.groups import Groups
from anemoi.datasets.use.gridded.misc import as_first_date
from anemoi.datasets.use.gridded.misc import as_last_date

LOG = logging.getLogger(__name__)


def _json_tidy(o: Any) -> Any:
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


def _build_statistics_dates(
    dates: list[datetime.datetime],
    start: datetime.datetime | None,
    end: datetime.datetime | None,
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
        from anemoi.datasets.create.zarr import add_zarr_dataset

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
            z.attrs[k] = json.loads(json.dumps(v, default=_json_tidy))

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


class FieldTaskMixin(Task):
    """A base class for dataset creation tasks."""

    dataset_class = WritableDataset

    @cached_property
    def dataset(self) -> WritableDataset:
        self.dataset = self.dataset_class(self.path)

    def update_metadata(self, **kwargs: Any) -> None:
        """Update the metadata of the dataset.

        Parameters
        ----------
        **kwargs
            The metadata to update.
        """
        self.dataset.update_metadata(**kwargs)

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


class HasRegistryMixin:
    """A mixin class to provide registry functionality."""

    @cached_property
    def registry(self) -> Any:
        """Get the registry."""
        from anemoi.datasets.create.zarr import ZarrBuiltRegistry

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

        self.context = FieldContext(
            order_by=self.output.order_by,
            flatten_grid=self.output.flatten_grid,
            remapping=build_remapping(self.output.remapping),
            use_grib_paramid=config.build.use_grib_paramid,
        )

        self.input = InputBuilder(
            config.input,
            data_sources=config.get("data_sources", {}),
        )
        LOG.debug("✅ INPUT_BUILDER")
        LOG.debug(self.input)


def validate_config(config: Any) -> None:

    import json

    import jsonschema

    def _tidy(d):
        if isinstance(d, dict):
            return {k: _tidy(v) for k, v in d.items()}

        if isinstance(d, list):
            return [_tidy(v) for v in d if v is not None]

        # jsonschema does not support datetime.date
        if isinstance(d, datetime.datetime):
            return d.isoformat()

        if isinstance(d, datetime.date):
            return d.isoformat()

        return d

    # https://json-schema.org

    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "schemas",
            "recipe.json",
        )
    ) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=_tidy(config), schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        LOG.error("❌ Config validation failed (jsonschema):")
        LOG.error(e.message)
        raise
