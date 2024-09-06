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

    @cached_property
    def registry(self):
        return ZarrBuiltRegistry(self.path, use_threads=self.use_threads)

    def ready(self):
        return all(self.registry.get_flags())


class GenericAdditions(GenericDatasetHandler):

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