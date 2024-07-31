# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
from functools import cached_property

import numpy as np
from obsdata.dataset.obs_dataset import ObsDataset

from anemoi.datasets.data.misc import _frequency_to_hours

from ..debug import Node
from ..stores import zarr_lookup

LOG = logging.getLogger(__name__)


def _resolve_path(path):
    return zarr_lookup(path)


class ObservationsBase:
    def mutate(self):
        return self

    def _check(self):
        pass

    def __len__(self):
        return len(self.dates)

    def tree(self):
        return Node(self)

    def _subset(self, *args, **kwargs):
        return self

    def __getitem__(self, i):
        # if isinstance(i, slice):
        #    return [self.getitem(j) for j in range(int(slice.start), int(slice.stop))]
        if isinstance(i, int):
            return self.getitem(i)
        # if isinstance(i, list):
        #    return [self.getitem(j) for j in i]
        raise TypeError(f"Expected int, got {type(i)}")

    @cached_property
    def dates(self):
        delta = datetime.timedelta(hours=self.frequency)
        dates = []
        current_date = self.start_date
        while current_date <= self.end_date:
            dates.append(current_date)
            current_date += delta
        return dates


class Dictionary(ObservationsBase):
    def __init__(self, datasets):
        _datasets = list(datasets.values())
        self.frequency = _datasets[0].frequency
        for d in _datasets[1:]:
            assert d.frequency == self.frequency, f"Expected {self.frequency}, got {d.frequency}"

        self.start_date = min(d.start_date for d in _datasets)
        self.end_date = max(d.end_date for d in _datasets)

        self.datasets = {k: Padded(d, self.start_date, self.end_date).mutate() for k, d in datasets.items()}

    def getitem(self, i):
        item = {k: d[i] for k, d in self.datasets.items()}
        return {k: v for k, v in item.items() if v is not None}

    def tree(self):
        return Node(self, [d.tree() for k, d in self.datasets.items()])


class Padded(ObservationsBase):
    def __init__(self, dataset, start, end):
        self.forward = dataset.mutate()
        self.frequency = self.forward.frequency
        self.start_date = start
        self.end_date = end

    def getitem(self, i):
        date = self.dates[i]
        for j, d in enumerate(self.forward.dates):
            if date == d:
                return self.forward[j]
        return None

    def tree(self):
        return Node(
            self,
            [self.forward.tree()],
            frequency=self.frequency,
            start=self.start_date,
            end=self.end_date,
        )


class Observations(ObservationsBase):
    def __init__(self, dataset, start, end, frequency, time_span=None):
        assert not dataset.endswith(".zarr"), f"Expected dataset name, got {dataset}"
        self.frequency = _frequency_to_hours(frequency)
        self.time_span = time_span  # not used
        self.path = _resolve_path(dataset)
        self.start_date = start
        self.end_date = end

        # _start_date must be the begginning of the time window of the first item
        _start_date = (self.start_date - datetime.timedelta(hours=self.frequency)).strftime("%Y%m%d%H%M%S")
        _start_date = int(_start_date)
        # _end_date must be the end of the time window of the last item
        _end_date = int(self.end_date.strftime("%Y%m%d%H%M%S"))

        self.forward = ObsDataset(
            self.path,
            _start_date,
            _end_date,
            len_hrs=self.frequency,  # length the time windows, i.e. the time span of one item
            step_hrs=self.frequency,  # frequency of the dataset, i.e. the time shift between two items
            normalize=False,
        )

        assert self.frequency == self.forward.step_hrs, f"Expected {self.frequency}, got {self.forward.len_hrs}"
        assert self.frequency == self.forward.len_hrs, f"Expected {self.frequency}, got {self.forward.step_hrs}"

        if len(self.forward) != len(self.dates):
            raise ValueError(
                (
                    f"Dates are not consistent with the number of items in the dataset. "
                    f"The dataset contains {len(self.forward)} time windows. "
                    f"This is not compatible with what is requested: "
                    f"{len(self.dates)} are requested from {self.start_date} to {self.end_date} "
                    f"with frequency={self.frequency}."
                )
            )

    def getitem(self, i):
        ##########################
        # TODO when the forward is ready
        #    end = self.dates[i]
        #    start = end - datetime.timedelta(hours=self.frequency)
        #    # this should get directly the numpy array
        #    data = self.forward.get_data_from_dates_interval(start, end)
        data = self.forward[i]
        data = data.numpy().astype(np.float32)
        ##########################

        if data.shape[0] == 0:
            return None
        else:
            return data

    def tree(self):
        return Node(
            self,
            [],
            path=self.path,
            frequency=self.frequency,
            START=self.start_date,
            END=self.end_date,
        )

    def __repr__(self):
        return f"Observations({os.path.basename(self.path)}, {self.dates[0]};{self.dates[-1]}, {len(self)})"


def _open(a):
    if isinstance(a, ObservationsBase):
        return a.mutate()
    if isinstance(a, dict):
        return _open_observations(**a)
    # if isinstance(a, str):
    #     return Observations(a)
    raise NotImplementedError(f"Expected ObservationsBase or dict, got {type(a)}")


def observations_factory(args, kwargs):
    cfg = kwargs.pop("observations")
    assert len(args) == 0, args
    assert len(kwargs) == 0, kwargs
    assert isinstance(cfg, dict), type(cfg)

    return _open_observations(**cfg)


def _open_observations(*args, **kwargs):
    if "pad" in kwargs:
        assert len(args) == 0
        pad = kwargs.pop("pad")
        dataset = _open(pad).mutate()
        return Padded(dataset, **kwargs).mutate()

    if "dictionary" in kwargs:
        assert len(args) == 0
        dictionary = kwargs.pop("dictionary")
        datasets = {k: _open(d).mutate() for k, d in dictionary.items()}
        return Dictionary(datasets).mutate()

    assert len(args) == 0, args
    for k, v in kwargs.items():
        assert k in ["dataset", "start", "end", "frequency", "time_span"], k
    return Observations(*args, **kwargs).mutate()
