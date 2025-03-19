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
from abc import abstractmethod
from collections import defaultdict
from functools import cached_property

import numpy as np
import yaml
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.data import open_dataset

LOG = logging.getLogger(__name__)

if os.environ.get("ANEMOI_DATASET_COUNTER", "0") == "1":

    def counter(func):
        def wrapper(*args, **kwargs):
            count = 0
            for i in range(len(args[0])):
                count += 1
                yield func(*args, **kwargs)
            print(f"Counter: {count} calls to {func.__name__}")

        return wrapper

else:

    def counter(func):
        return func


def open_multi_datasets(*datasets, **kwargs):

    if len(datasets) == 1 and datasets[0].endswith(".vz"):
        return open_vs_dataset(datasets[0], **kwargs)

    for d in datasets:
        assert not d.endswith(".vz"), f"mixing datasets type not implemented yet. {datasets}"

    return LegacyDatasets(datasets, **kwargs)


def open_vs_dataset(dataset, **kwargs):
    if not dataset.endswith(".vz"):
        raise ValueError("dataset must be a .vz file")
    return VzDatasets(dataset, **kwargs)


class LegacyDatasets:
    def __init__(self, paths, start=None, end=None, **kwargs):
        self.paths = paths

        if not start or not end:
            print(
                "❌❌ Warning: start and end not provided, using the minima first and maximal last dates of the datasets"
            )
            lst = [self._open_dataset(p, **kwargs) for p in paths]
            start = min([d.dates[0] for d in lst])
            end = max([d.dates[-1] for d in lst])

        self._datasets = {
            os.path.basename(p).split(".")[0]: self._open_dataset(p, start=start, end=end, padding="empty")
            for p in paths
        }

        first = list(self._datasets.values())[0]
        for name, dataset in self._datasets.items():
            if dataset.dates[0] != first.dates[0] or dataset.dates[-1] != first.dates[-1]:
                raise ValueError("Datasets have different start and end times")
            if dataset.frequency != first.frequency:
                raise ValueError("Datasets have different frequencies")

        self._keys = self._datasets.keys

        self._first = list(self._datasets.values())[0]

    def _open_dataset(self, p, **kwargs):
        if p.startswith("observations-"):
            return open_dataset(observations=p, **kwargs)
        else:
            print("❗ Opening non-observations dataset:", p)
            return open_dataset(p, **kwargs)

    def items(self):
        return self._datasets.items()

    @property
    def dates(self):
        return self._first.dates

    def __len__(self):
        return len(self._first)

    def __getitem__(self, i):
        return {k: d[i] for k, d in self._datasets.items()}


class DictDataset:
    def __call__(self, i=None, group=None):
        if group is None and i is not None:
            return LazyMultiElement(self, i)

        if group is not None and i is None:
            return OneGroup(self, group)

        if group is not None and i is not None:
            return LazyMultiElement(self, i)[group]

        raise ValueError("Either i or group must be provided, both are None")

    @abstractmethod
    def __getitem__(self, i):
        pass

    @property
    def start_date(self):
        return self.dates[0]

    @property
    def end_date(self):
        return self.dates[-1]

    def _subset(self, **kwargs):
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        frequency = kwargs.pop("frequency", self.frequency)

        if frequency != self.frequency:
            raise ValueError(f"Changing the frequency {frequency} (from {self.frequency}) is not implemented yet.")

        if start is not None or end is not None:

            def _dates_to_indices(start, end):
                from .misc import as_first_date
                from .misc import as_last_date

                start = self.dates[0] if start is None else as_first_date(start, self.dates)
                end = self.dates[-1] if end is None else as_last_date(end, self.dates)

                return [i for i, date in enumerate(self.dates) if start <= date <= end]

            return Subset(
                self, _dates_to_indices(start, end), {"start": start, "end": end, "frequency": frequency}
            )._subset(**kwargs)

        select = kwargs.pop("select", None)
        if select is not None:
            return Select(self, select)._subset(**kwargs)

        return self

    def mutate(self):
        return self

    def _check(self):
        pass

    @property
    def variables_dict(self):
        dic = defaultdict(list)
        for v in self.variables:
            if v in dic:
                raise ValueError(f"Duplicate variable name {v} in {self.variables}")
            k, v = v.split(".")
            dic[k].append(v)
        return dic

    @property
    def name_to_index_dict(self):
        dic = defaultdict(dict)
        for name, i in self.name_to_index.items():
            group, k = name.split(".")
            dic[group][k] = i
        return dic


class Forward(DictDataset):
    def __init__(self, dataset):
        self.forward = dataset

    @property
    def statistics(self):
        return self.forward.statistics

    @property
    def name_to_index(self):
        return self.forward.name_to_index

    @property
    def variables(self):
        return self.forward.variables

    def keys(self):
        return self.forward.keys()

    @property
    def dates(self):
        return self.forward.dates

    @property
    def frequency(self):
        return self.forward.frequency

    @property
    def shapes(self):
        return self.forward.shapes


def match_variable(lst, name, group=None):
    if isinstance(lst, dict):
        lst = [f"{k}.{v}" for k, v in lst.items()]

    if not isinstance(lst, list):
        raise ValueError(f"Expecting a list or dict, not {type(lst)}")
    for v in lst:
        if not isinstance(v, str):
            raise ValueError(f"Expecting a list of strings, not '{v}' : {type(v)}")

    if name in lst:
        return True
    if f"*.{name}" in lst:
        return True

    if group is not None:
        if f"{group}.{name}" in lst:
            return True
        if f"{group}.*" in lst:
            return True
    return False


class Select(Forward):
    def __init__(self, dataset, select):
        super().__init__(dataset)

        self.dataset = dataset
        self._select = select

        self.reason = {"select": select}
        self._build_indices_and_name_to_index()

    def _build_indices_and_name_to_index(self):
        self._indices = {}
        self._name_to_index = {}
        for group in self.dataset.keys():
            variables = self.dataset.variables_dict[group]
            ind = np.zeros(len(variables), dtype=bool)
            count = 0
            for j, v in enumerate(variables):
                if self.match_variable(v, group):
                    ind[variables.index(v)] = True
                    self._indices[group] = ind
                    self._name_to_index[f"{group}.{v}"] = (group, count)
                    count += 1
            assert np.sum(ind) == count, f"Mismatch in {group}: {variables}, {ind}"

    def match_variable(self, name, group):
        return match_variable(self._select, name, group)

    def keys(self):
        return [k for k in self.dataset.keys() if k in self._indices]

    def __getitem__(self, i):
        dic = self.dataset[i]
        return {k: v[self._indices[k]] for k, v in dic.items() if k in self._indices}

    @cached_property
    def name_to_index(self):
        return self._name_to_index

    @property
    def statistics(self):
        dic = {}
        for key, values in self.dataset.statistics.items():
            dic[key] = {k: v[self._indices[k]] for k, v in values.items() if k in self._indices}
        return dic


class Subset(Forward):
    def __init__(self, dataset, indices, reason):
        super().__init__(dataset)

        self.dataset = dataset
        self._indices = indices

        self.reason = reason
        self._dates = dataset.dates[indices]

    def dates(self):
        return self._dates

    def __getitem__(self, i):
        return self.dataset[self._indices[i]]

    def __len__(self):
        return len(self._indices)


class VzDatasets(DictDataset):
    _metadata = None

    def __init__(self, path, **kwargs):
        if kwargs:
            print("Warning: ignoring kwargs", kwargs)
        self.path = path
        self.keys = self.metadata["sources"].keys

    @property
    def frequency(self):
        frequency = self.metadata["frequency"]
        frequency = frequency_to_timedelta(frequency)
        return frequency

    @property
    def name_to_index(self):
        return self.metadata["name_to_index"]

    @property
    def variables(self):
        return self.metadata["variables"]

    @property
    def metadata(self):
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    @property
    def shapes(self):
        return self.metadata["shapes"]

    def items(self, *args, **kwargs):
        return {k: OneGroup(self, k) for k in self.keys()}.items(*args, **kwargs)

    @cached_property
    def statistics(self):
        dirpath = os.path.join(self.path, "statistics")
        statistics = {}
        for p in os.listdir(dirpath):
            key = os.path.basename(p).split(".")[0]
            path = os.path.join(dirpath, p)
            statistics[key] = dict(np.load(path))
        return statistics

    def _load_metadata(self):
        if os.path.exists(os.path.join(self.path, "metadata.json")):
            with open(os.path.join(self.path, "metadata.json"), "r") as f:
                self._metadata = json.load(f)
            return
        with open(os.path.join(self.path, "metadata.yaml"), "r") as f:
            self._metadata = yaml.safe_load(f)

    def __len__(self):
        return len(self.dates)

    @property
    def start_date(self):
        date = self._metadata["start_date"]
        return datetime.datetime.fromisoformat(date)

    @property
    def end_date(self):
        date = self._metadata["end_date"]
        return datetime.datetime.fromisoformat(date)

    @cached_property
    def dates(self):
        result = []
        delta = self.frequency
        d = self.start_date
        while d <= self.end_date:
            result.append(d)
            d += delta
        return np.array(result)

    @counter
    def __getitem__(self, i):
        path = os.path.join(self.path, "data", str(int(i / 10)), f"{i}.npz")
        with open(path, "rb") as f:
            return dict(np.load(f))


def find_latitude(name_to_index, k):
    for name, index in name_to_index.items():
        if name == f"{k}.latitude":
            return index[1]
        if name == f"{k}.lat":
            return index[1]
    raise ValueError(f"No latitude found in name_to_index: {list(name_to_index.keys())}, {k}")


def find_longitude(name_to_index, k):
    for name, index in name_to_index.items():
        if name == f"{k}.longitude":
            return index[1]
        if name == f"{k}.lon":
            return index[1]
    raise ValueError(f"No longitude found in name_to_index: {list(name_to_index.keys())}, {k}")


class LazyMultiElement:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n
        self.latitudes_indexes = {k: find_latitude(dataset.name_to_index, k) for k in dataset.keys()}
        self.longitudes_indexes = {k: find_longitude(dataset.name_to_index, k) for k in dataset.keys()}

    @cached_property
    def values(self):
        return self.dataset[self.n]

    def keys(self):
        return self.dataset.keys()

    def __getitem__(self, key):
        return self.values[key]

    def items(self):
        return self.values.items()

    @property
    def name_to_index(self):
        return self.dataset.name_to_index

    @property
    def name_to_index_dict(self):
        return self.dataset.name_to_index_dict

    @property
    def statistics(self):
        return self.dataset.statistics

    @property
    def latitudes(self):
        return {k: self.values[k][v, ...] for k, v in self.latitudes_indexes.items()}

    @property
    def longitudes(self):
        return {k: self.values[k][v, ...] for k, v in self.longitudes_indexes.items()}


class OneGroup:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def __getitem__(self, i):
        return self.dataset[i][self.name]

    @property
    def statistics(self):
        return {k: v[self.name] for k, v in self.dataset.statistics.items()}
