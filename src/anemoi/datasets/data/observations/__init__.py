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

from anemoi.datasets.data.misc import _frequency_to_hours

from ..debug import Node
from ..stores import zarr_lookup

LOG = logging.getLogger(__name__)


def _resolve_path(path):
    return zarr_lookup(path)


def make_dates(start, end, frequency):
    if isinstance(start, np.datetime64):
        start = start.astype(datetime.datetime)
    if isinstance(end, np.datetime64):
        end = end.astype(datetime.datetime)

    delta = datetime.timedelta(hours=frequency)
    dates = []
    current_date = start
    while current_date <= end:
        dates.append(current_date)
        current_date += delta
    return dates


def merge_dates(datasets):
    start_date = None
    end_date = None
    for d in datasets:
        s, e = min(d.dates), max(d.dates)
        if start_date is None or s < start_date:
            start_date = s
        if end_date is None or e > end_date:
            end_date = e
    return start_date, end_date


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

    @property
    def variables(self):
        raise NotImplementedError()


class Multiple(ObservationsBase):
    def __init__(self, datasets):
        _datasets = list(datasets.values())
        self.frequency = _datasets[0].frequency
        for d in _datasets[1:]:
            assert d.frequency == self.frequency, f"Expected {self.frequency}, got {d.frequency}"

        start_date, end_date = merge_dates(_datasets)

        self.datasets = {k: Padded(d, start_date, end_date).mutate() for k, d in datasets.items()}
        self.dates = make_dates(start_date, end_date, self.frequency)

    def getitem(self, i):
        item = {k: d[i] for k, d in self.datasets.items()}
        return {k: v for k, v in item.items() if v is not None}

    def tree(self):
        return Node(self, [d.tree() for k, d in self.datasets.items()])

    def _check(self):
        names_dict = {k: d.variables for k, d in self.datasets.items()}

        names = []
        for _, v in names_dict.items():
            names += list(v)

        if len(set(names)) != len(names):
            # Found duplicated names
            msg = ""
            for name in set(names):
                if names.count(name) > 1:
                    for k, v in names_dict.items():
                        if name in v:
                            msg += f"Variable '{name}' is found in dataset '{k}'. "
                    break
            raise ValueError(f"Duplicated variable: {msg}")

    @property
    def variables(self):
        variables = []
        for k, d in self.datasets.items():
            variables += list(d.variables)
        return variables

    @cached_property
    def name_to_index(self):
        dic = {}
        for k, d in self.datasets.items():
            for name in d.variables:
                dic[name] = (k, d.name_to_index[name])
        return dic

    @property
    def statistics(self):
        return {k: v.statistics for k, v in self.datasets.items()}


class Forward(ObservationsBase):
    def __init__(self, dataset):
        self.forward = dataset.mutate()
        self.dates = self.forward.dates

    def tree(self):
        return Node(self, [self.forward.tree()])

    @property
    def variables(self):
        return self.forward.variables

    def __repr__(self):
        return f"Forward({self.forward})"

    def getitem(self, i):
        return self.forward[i]

    @property
    def frequency(self):
        return self.forward.frequency

    @cached_property
    def name_to_index(self):
        return {k: i for i, k in enumerate(self.variables)}

    @cached_property
    def statistics(self):
        return self.forward.statistics


class RenamePrefix(Forward):
    def __init__(self, dataset, prefix):
        super().__init__(dataset)
        self.prefix = prefix
        self._variables = [f"{prefix}_{n}" for n in self.forward.variables]

    @property
    def variables(self):
        return self._variables

    def tree(self):
        return Node(self, [self.forward.tree()], rename_prefix=self.prefix)


class Padded(Forward):
    def __init__(self, dataset, start, end):
        super().__init__(dataset)
        self._frequency = self.forward.frequency
        self._start_date = start
        self._end_date = end
        self.dates = make_dates(start, end, self._frequency)

    @property
    def frequency(self):
        return self._frequency

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
            start=self._start_date,
            end=self._end_date,
        )


class Observations(ObservationsBase):
    def __init__(self, dataset, start, end, frequency, time_span=None):
        assert not dataset.endswith(".zarr"), f"Expected dataset name, got {dataset}"
        self.frequency = _frequency_to_hours(frequency)
        self.time_span = time_span  # not used
        self.path = _resolve_path(dataset)
        self.dates = make_dates(start, end, self.frequency)
        self._start_date = start
        self._end_date = end

        # _start_date must be the begginning of the time window of the first item
        first_window_begin = (self._start_date - datetime.timedelta(hours=self.frequency)).strftime("%Y%m%d%H%M%S")
        first_window_begin = int(first_window_begin)
        # last_window_end must be the end of the time window of the last item
        last_window_end = int(self._end_date.strftime("%Y%m%d%H%M%S"))

        from obsdata.dataset.obs_dataset import ObsDataset

        self.forward = ObsDataset(
            self.path,
            first_window_begin,
            last_window_end,
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
                    f"{len(self.dates)} are requested from {self._start_date} to {self._end_date} "
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
        ##########################
        data = data.numpy().astype(np.float32)
        assert len(data.shape) == 2, f"Expected 2D array, got {data.shape}"
        data = data.T
        # insert an additional dimension of size 1 to have a layout similar to fields datasets (a, b) -> (a, 1, b)
        data = np.expand_dims(data, axis=1)

        if data.shape[0] == 0:
            return None
        else:
            return data

    @property
    def variables(self):
        colnames = self.forward.colnames
        variables = []
        for n in colnames:
            if n.startswith("obsvalue_"):
                n = n.replace("obsvalue_", "")
            variables.append(n)
        return variables

    @property
    def statistics(self):
        return StatisticsOfObsDataset(self.forward)

    def tree(self):
        return Node(
            self,
            [],
            path=self.path,
            frequency=self.frequency,
            START=self._start_date,
            END=self._end_date,
        )

    def __repr__(self):
        return f"Observations({os.path.basename(self.path)}, {self.dates[0]};{self.dates[-1]}, {len(self)})"


def _open(a):
    if isinstance(a, ObservationsBase):
        return a.mutate()
    if isinstance(a, dict):
        return _open_observations(**a).mutate()
    if isinstance(a, str):
        return _open_observations(a).mutate()
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

    if "multiple" in kwargs:
        assert len(args) == 0
        multiple = kwargs.pop("multiple")
        datasets = {k: _open(d).mutate() for k, d in multiple.items()}
        return Multiple(datasets).mutate()

    if "rename_prefix" in kwargs:
        prefix = kwargs.pop("rename_prefix")
        dataset = _open(kwargs).mutate()
        return RenamePrefix(dataset, prefix).mutate()

    if "is_observations" in kwargs:
        kwargs.pop("is_observations")
        assert len(args) == 0, args
        return Observations(*args, **kwargs).mutate()

    from ..misc import _open_dataset as _open_fields

    return _open_fields(*args, **kwargs).mutate()


class StatisticsOfObsDataset:
    def __init__(self, dataset):
        self.dataset = dataset
