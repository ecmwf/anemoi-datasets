# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
import datetime
import logging
import os
from functools import cached_property

import numpy as np
from anemoi.utils.dates import frequency_to_string as frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

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

    dates = []
    current_date = start
    while current_date <= end:
        dates.append(current_date)
        current_date += frequency
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
    resolution = None

    @cached_property
    def shape(self):
        return (len(self.dates), len(self.variables), "dynamic")

    def empty_item(self):
        return np.full(self.shape[1:-1] + (0,), 0.0, dtype=np.float32)

    def metadata(self):
        return dict(observations_datasets="obs datasets currenty have no metadata")

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
    def __init__(self, datasets, names=None):
        self._names = names
        self.frequency = datasets[0].frequency
        for d in datasets[1:]:
            assert d.frequency == self.frequency, f"Expected {self.frequency}, got {d.frequency}"

        start_date, end_date = merge_dates(datasets)

        self.datasets = [Padded(d, start_date, end_date).mutate() for d in datasets]
        self.dates = make_dates(start_date, end_date, self.frequency)

        # todo: implement missing
        assert all(
            d.missing == set() for d in self.datasets
        ), f"Expected no missing, got {[d.missing for d in self.datasets]}"
        self.missing = set()

    @property
    def names(self):
        return self._names

    def getitem(self, i):
        return [d[i] for d in self.datasets]

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])

    def _check(self):
        names = []
        for ds in self.datasets:
            for name in ds.variables:
                if name in names:
                    raise ValueError(f"Duplicated variable: {name}. Use rename_prefix to avoid this issue.")

    @property
    def variables(self):
        variables = []
        for ds in self.datasets:
            variables += ds.variables
        return variables

    @cached_property
    def name_to_index(self):
        dic = {}
        for i, d in enumerate(self.datasets):
            for name in d.variables:
                dic[name] = (i, d.name_to_index[name])
        return dic

    @property
    def statistics(self):
        keys = self.datasets[0].statistics.keys()
        dic = defaultdict(list)
        for d in self.datasets:
            for k in keys:
                dic[k].append(d.statistics[k])
        assert "mean" in dic, f"Expected 'mean' in statistics, got {list(dic.keys())}"
        return dic


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
        return f"{self.__class__.__name__}({self.forward})"

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
        self._variables = [f"{prefix}{n}" for n in self.forward.variables]
        for n in self._variables:
            if "-" in n and "_" in n:
                raise ValueError(f"Do nor mix '-' and '_', got {n}")

    @property
    def variables(self):
        return self._variables

    def tree(self):
        return Node(self, [self.forward.tree()], rename_prefix=self.prefix)


class Select(Forward):
    def __init__(self, dataset, select):
        super().__init__(dataset)
        self._select = select
        assert all(n in self.forward.variables for n in select), f"Expected {select} in {self.forward.variables}"

        self._variables = select
        self._indexes = tuple(self.forward.name_to_index[n] for n in self._variables)

    def tree(self):
        return Node(self, [self.forward.tree()], select=self._select)

    def getitem(self, i):
        data = self.forward[i]
        data = data[self._indexes,:]
        return data

    @property
    def statistics(self):
        dic = {}
        for k, data in self.forward.statistics.items():
            assert len(data.shape) == 1, f"Expected 1D array, got {data.shape}"
            dic[k] = data[self._indexes,] # notice the "," here because this 1D array is indexed using a tuple.
        return dic

    @property
    def variables(self):
        return self._variables

    @property
    def name_to_index(self):
        return {n:i for i, n in enumerate(self._variables)}


class Padded(Forward):
    def __init__(self, dataset, start, end):
        super().__init__(dataset)
        self._frequency = self.forward.frequency
        assert isinstance(self._frequency, datetime.timedelta), f"Expected timedelta, got {type(self._frequency)}"
        self._start_date = start
        self._end_date = end
        self.dates = make_dates(start, end, self._frequency)

    @property
    def missing(self):
        return set()

    @property
    def frequency(self):
        return self._frequency

    def getitem(self, i):
        # TODO: very inefficient, improve this
        date = self.dates[i]
        for j, d in enumerate(self.forward.dates):
            if date == d:
                return self.forward[j]
        return self.empty_item()

    def tree(self):
        return Node(
            self,
            [self.forward.tree()],
            frequency=self.frequency,
            start=self._start_date,
            end=self._end_date,
        )


def is_observations_dataset(path):
    import zarr

    z = zarr.open(path, mode="r")
    try:
        return z.data.attrs["is_observations"] is True
    except:  # noqa
        return False


def round_datetime(dt, frequency, up=True):
    dt = dt.replace(minute=0, second=0, microsecond=0)
    hour = dt.hour
    if hour % frequency != 0:
        dt = dt.replace(hour=(hour // frequency) * frequency)
        dt = dt + datetime.timedelta(hours=frequency)
    return dt


class Observations(ObservationsBase):
    def __init__(self, dataset, frequency, window=None):
        assert not dataset.endswith(".zarr"), f"Expected dataset name, got {dataset}"
        self.frequency = frequency_to_timedelta(frequency)
        assert self.frequency.total_seconds() % 3600 == 0, f"Expected multiple of 3600, got {self.frequency}"

        frequency_hours = int(self.frequency.total_seconds() // 3600)
        assert isinstance(frequency_hours, int), f"Expected int, got {type(frequency_hours)}"

        if window is None:
            window = (-frequency_hours, 0)
        if window == (-frequency_hours, 0):
            raise ValueError("For now, only window = (- frequency, 0) are supported")

        self.window = window
        self.path = _resolve_path(dataset)
        assert is_observations_dataset(self.path), f"Expected observations dataset, got {self.path}"

        start, end = self._probe_attributes["start_date"], self._probe_attributes["end_date"]
        # print(f"✅ from attribute start={start}, end={end}")
        start, end = datetime.datetime.fromisoformat(start), datetime.datetime.fromisoformat(end)
        start, end = round_datetime(start, frequency_hours), round_datetime(end, frequency_hours)
        # print(f"       rounded to start={start}, end={end}")

        self.dates = make_dates(start + self.frequency, end, self.frequency)
        # print(f"              -> dates: {self.dates[0]}, {self.dates[-1]}")
        # print(f"                   nb of dates: {len(self.dates)}")

        first_window_begin = start.strftime("%Y%m%d%H%M%S")
        first_window_begin = int(first_window_begin)
        # last_window_end must be the end of the time window of the last item
        last_window_end = int(end.strftime("%Y%m%d%H%M%S"))

        from obsdata.dataset.obs_dataset import ObsDataset

        args = [self.path, first_window_begin, last_window_end]
        kwargs = dict(
            len_hrs=frequency_hours,  # length the time windows, i.e. the time span of one item
            step_hrs=frequency_hours,  # frequency of the dataset, i.e. the time shift between two items
            normalize=False,
        )
        self.forward = ObsDataset(*args, **kwargs)
        print(f"TRACE: ObsDataset({args}, {kwargs})")

        # print(f"len(obs)={len(self.forward)}")

        assert frequency_hours == self.forward.step_hrs, f"Expected {frequency_hours}, got {self.forward.len_hrs}"
        assert frequency_hours == self.forward.len_hrs, f"Expected {frequency_hours}, got {self.forward.step_hrs}"

        if len(self.forward) != len(self.dates):
            raise ValueError(
                (
                    f"Dates are not consistent with the number of items in the dataset. "
                    f"The dataset contains {len(self.forward)} time windows. "
                    f"This is not compatible with the "
                    f"{len(self.dates)} requested dates with frequency={frequency_hours}"
                    f"{self.dates[0]}, {self.dates[1]}, ..., {self.dates[-2]}, {self.dates[-1]} "
                )
            )

    @cached_property
    def _probe_attributes(self):
        import zarr

        z = zarr.open(self.path, mode="r")
        return dict(z.data.attrs)

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

        if not data.size:
            data = self.empty_item()
        assert data.shape[0] == self.shape[1], f"Data shape {data.shape} does not match {self.shape}"
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
        mean = self.forward.properties["means"]
        mean = np.array(mean, dtype=np.float32)
        var = self.forward.properties["vars"]
        var = np.array(var, dtype=np.float32)
        stdev = np.sqrt(var)
        minimum = np.full_like(mean, np.nan)
        maximum = np.full_like(mean, np.nan)
        print(f"✅Cannot find minimum using np.nan")
        print(f"✅Cannot find maximum using np.nan")

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
    if "select" in kwargs:
        select = kwargs.pop("select")
        dataset = _open_observations(*args, **kwargs).mutate()
        return Select(dataset, select=select).mutate()

    if "pad" in kwargs:
        assert len(args) == 0
        pad = kwargs.pop("pad")
        dataset = _open(pad).mutate()
        return Padded(dataset, **kwargs).mutate()

    if "multiple" in kwargs:
        assert len(args) == 0
        multiple = kwargs.pop("multiple")
        datasets = [_open(d).mutate() for d in multiple]
        return Multiple(datasets, **kwargs).mutate()

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
