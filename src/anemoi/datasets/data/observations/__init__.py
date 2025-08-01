# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
from functools import cached_property
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.data.dataset import Dataset

from ..debug import Node

LOG = logging.getLogger(__name__)


def round_datetime(dt, frequency, up=True):
    dt = dt.replace(minute=0, second=0, microsecond=0)
    hour = dt.hour
    if hour % frequency != 0:
        dt = dt.replace(hour=(hour // frequency) * frequency)
        dt = dt + datetime.timedelta(hours=frequency)
    return dt


def make_dates(start, end, frequency):
    if isinstance(start, np.datetime64):
        start = start.astype(datetime.datetime)
    if isinstance(end, np.datetime64):
        end = end.astype(datetime.datetime)

    dates = []
    current_date = start
    while current_date <= end:
        dates.append(current_date)
        current_date += frequency

    dates = [np.datetime64(d, "s") for d in dates]
    dates = np.array(dates, dtype="datetime64[s]")
    return dates


class ObservationsBase(Dataset):
    resolution = None

    @cached_property
    def shape(self):
        return (len(self.dates), len(self.variables), "dynamic")

    def empty_item(self):
        return np.full(self.shape[1:-1] + (0,), 0.0, dtype=np.float32)

    def metadata(self):
        return dict(observations_datasets="obs datasets currenty have no metadata")

    def _check(self):
        pass

    def __len__(self):
        return len(self.dates)

    def tree(self):
        return Node(self)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.getitem(i)

        # The following may would work but is likely to change in the future
        # if isinstance(i, slice):
        #    return [self.getitem(j) for j in range(int(slice.start), int(slice.stop))]
        # if isinstance(i, list):
        #    return [self.getitem(j) for j in i]

        raise ValueError(
            f"Expected int, got {i} of type {type(i)}. Only int is supported to index "
            "observations datasets. Please use a second [] to select part of the data [i][a,b,c]"
        )

    @property
    def variables(self):
        raise NotImplementedError()

    def collect_input_sources(self):
        LOG.warning("collect_input_sources method is not implemented")
        return []

    def constant_fields(self):
        LOG.warning("constant_fields method is not implemented")
        return []

    @property
    def dates(self):
        return self._dates

    @property
    def dtype(self):
        return np.float32

    @property
    def field_shape(self):
        return self.shape[1:]

    @property
    def frequency(self):
        assert isinstance(self._frequency, datetime.timedelta), f"Expected timedelta, got {type(self._frequency)}"
        return self._frequency

    @property
    def latitudes(self):
        raise NotImplementedError("latitudes property is not implemented")

    @property
    def longitudes(self):
        raise NotImplementedError("longitudes property is not implemented")

    @property
    def missing(self):
        return []

    def statistics_tendencies(self):
        raise NotImplementedError("statistics_tendencies method is not implemented")

    def variables_metadata(self):
        raise NotImplementedError("variables_metadata method is not implemented")


class ObservationsZarr(ObservationsBase):
    def __init__(self, dataset, frequency=None, window=None):
        import zarr

        if isinstance(dataset, zarr.hierarchy.Group):
            dataset = dataset._store.path

        from ..stores import zarr_lookup

        dataset = zarr_lookup(dataset)
        self.path = dataset
        assert self._probe_attributes["is_observations"], f"Expected observations dataset, got {self.path}"

        if frequency is None:
            frequency = self._probe_attributes.get("frequency")
            # LOG.warning(f"Frequency not provided, using the one from the dataset: {frequency}")
        if frequency is None:
            frequency = "6h"
            # LOG.warning(f"Frequency not provided in the dataset, using the default : {frequency}")
        self._frequency = frequency_to_timedelta(frequency)
        assert self.frequency.total_seconds() % 3600 == 0, f"Expected multiple of 3600, got {self.frequency}"
        if self.frequency.total_seconds() != 6 * 3600:
            LOG.warning("Frequency is not 6h, this has not been tested, behaviour is unknown")

        frequency_hours = int(self.frequency.total_seconds() // 3600)
        assert isinstance(frequency_hours, int), f"Expected int, got {type(frequency_hours)}"

        if window is None:
            window = (-frequency_hours, 0)
        if window != (-frequency_hours, 0):
            raise ValueError("For now, only window = (- frequency, 0) are supported")

        self.window = window

        start, end = self._probe_attributes["start_date"], self._probe_attributes["end_date"]
        start, end = datetime.datetime.fromisoformat(start), datetime.datetime.fromisoformat(end)
        start, end = round_datetime(start, frequency_hours), round_datetime(end, frequency_hours)

        self._dates = make_dates(start + self.frequency, end, self.frequency)

        first_window_begin = start.strftime("%Y%m%d%H%M%S")
        first_window_begin = int(first_window_begin)
        # last_window_end must be the end of the time window of the last item
        last_window_end = int(end.strftime("%Y%m%d%H%M%S"))

        from .legacy_obs_dataset import ObsDataset

        args = [self.path, first_window_begin, last_window_end]
        kwargs = dict(
            len_hrs=frequency_hours,  # length the time windows, i.e. the time span of one item
            step_hrs=frequency_hours,  # frequency of the dataset, i.e. the time shift between two items
        )
        self.forward = ObsDataset(*args, **kwargs)

        assert frequency_hours == self.forward.step_hrs, f"Expected {frequency_hours}, got {self.forward.len_hrs}"
        assert frequency_hours == self.forward.len_hrs, f"Expected {frequency_hours}, got {self.forward.step_hrs}"

        if len(self.forward) != len(self.dates):
            raise ValueError(
                f"Dates are not consistent with the number of items in the dataset. "
                f"The dataset contains {len(self.forward)} time windows. "
                f"This is not compatible with the "
                f"{len(self.dates)} requested dates with frequency={frequency_hours}"
                f"{self.dates[0]}, {self.dates[1]}, ..., {self.dates[-2]}, {self.dates[-1]} "
            )

    @property
    def source(self):
        return self.path

    def get_dataset_names(self):
        name = os.path.basename(self.path)
        if name.endswith(".zarr"):
            name = name[:-5]
        return [name]

    @cached_property
    def _probe_attributes(self):
        import zarr

        z = zarr.open(self.path, mode="r")
        return dict(z.data.attrs)

    def get_aux(self, i):
        data = self.forward[i]

        latitudes = data[:, self.name_to_index["__latitudes"]].numpy()
        longitudes = data[:, self.name_to_index["__longitudes"]].numpy()

        reference = self.dates[i]
        times = self.forward.get_dates(i)
        if str(times.dtype) != "datetime64[s]":
            LOG.warning(f"Expected np.datetime64[s], got {times.dtype}. ")
            times = times.astype("datetime64[s]")
        assert str(reference.dtype) == "datetime64[s]", f"Expected np.datetime64[s], got {type(reference)}"
        timedeltas = times - reference

        assert latitudes.shape == longitudes.shape, f"Expected {latitudes.shape}, got {longitudes.shape}"
        assert timedeltas.shape == latitudes.shape, f"Expected {timedeltas.shape}, got {latitudes.shape}"

        return latitudes, longitudes, timedeltas

    def getitem(self, i):
        data = self.forward[i]

        data = data.numpy().astype(np.float32)
        assert len(data.shape) == 2, f"Expected 2D array, got {data.shape}"
        data = data.T

        if not data.size:
            data = self.empty_item()
        assert (
            data.shape[0] == self.shape[1]
        ), f"Data shape {data.shape} does not match {self.shape} :  {data.shape[0]} != {self.shape[1]}"
        return data

    @cached_property
    def variables(self):
        colnames = self.forward.colnames
        variables = []
        for n in colnames:
            if n.startswith("obsvalue_"):
                n = n.replace("obsvalue_", "")
            if n == "latitude" or n == "lat":
                assert "latitudes" not in variables, f"Duplicate latitudes found in {variables}"
                variables.append("__latitudes")
                continue
            if n == "longitude" or n == "lon":
                assert "longitudes" not in variables, f"Duplicate longitudes found in {variables}"
                variables.append("__longitudes")
                continue
            assert not n.startswith("__"), f"Invalid name {n} found in {colnames}"
            variables.append(n)
        return variables

    @property
    def name_to_index(self):
        return {n: i for i, n in enumerate(self.variables)}

    @property
    def statistics(self):
        mean = self.forward.properties["means"]
        mean = np.array(mean, dtype=np.float32)

        var = self.forward.properties["vars"]
        var = np.array(var, dtype=np.float32)
        stdev = np.sqrt(var)

        minimum = np.array(self.forward.z.data.attrs["mins"], dtype=np.float32)
        maximum = np.array(self.forward.z.data.attrs["maxs"], dtype=np.float32)

        assert isinstance(mean, np.ndarray), f"Expected np.ndarray, got {type(mean)}"
        assert isinstance(stdev, np.ndarray), f"Expected np.ndarray, got {type(stdev)}"
        assert isinstance(minimum, np.ndarray), f"Expected np.ndarray, got {type(minimum)}"
        assert isinstance(maximum, np.ndarray), f"Expected np.ndarray, got {type(maximum)}"
        return dict(mean=mean, stdev=stdev, minimum=minimum, maximum=maximum)

    def tree(self):
        return Node(
            self,
            [],
            path=self.path,
            frequency=self.frequency,
        )

    def __repr__(self):
        return f"Observations({os.path.basename(self.path)}, {self.dates[0]};{self.dates[-1]}, {len(self)})"


def observations_factory(args: tuple[Any, ...], kwargs: dict[str, Any]) -> ObservationsBase:
    observations = kwargs.pop("observations")

    if not isinstance(observations, dict):
        observations = dict(dataset=observations)
    dataset = ObservationsZarr(**observations)
    return dataset._subset(**kwargs)
