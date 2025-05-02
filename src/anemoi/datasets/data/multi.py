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
from collections import defaultdict
from functools import cached_property

import numpy as np
import yaml
from anemoi.utils.dates import frequency_to_timedelta

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
        return open_vz_dataset(datasets[0], **kwargs)

    for d in datasets:
        assert not d.endswith(".vz"), f"mixing datasets type not implemented yet. {datasets}"

    from anemoi.datasets.data.observations.multi import LegacyDatasets

    return LegacyDatasets(datasets, **kwargs)


def open_vz_dataset(dataset, **kwargs):
    if not dataset.endswith(".vz"):
        raise ValueError("dataset must be a .vz file")
    return VzDatasets(dataset, **kwargs)


class DictDataset:

    def __getitem__(self, i):
        if isinstance(i, str):
            return self._getgroup(i)

        if isinstance(i, int):
            return self._getelement(i)

        raise ValueError(f"Invalid index {i}, must be int or str")

    def _getgroup(self, i):
        return OneGroup(self, i)

    def _getelement(self, i):
        return LazyMultiElement(self, i)

    def _load_data(self, i):
        raise NotImplementedError("Must be implemented in subclass")

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
    def name_to_index(self):
        raise NotImplementedError("Must be implemented in subclass")

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
    def variables(self):
        return self.forward.variables

    def keys(self):
        return self.forward.keys()

    @property
    def dates(self):
        return self.forward.dates

    @property
    def name_to_index_dict(self):
        return self.forward.name_to_index_dict

    @property
    def name_to_index(self):
        return self.forward.name_to_index

    @property
    def frequency(self):
        return self.forward.frequency

    @property
    def shapes(self):
        return self.forward.shapes

    def __len__(self):
        return len(self.forward)


def match_variable(lst, name, group=None):
    # lst can be :
    # - a list of strings with dots
    # - a dict with keys as group and values as list of strings
    if isinstance(lst, dict):
        lst = [f"{k}.{v}" for k, v in lst.items()]

    if not isinstance(lst, list):
        raise ValueError(f"Expecting a list or dict, not {type(lst)}")
    for v in lst:
        if not isinstance(v, str):
            raise ValueError(f"Expecting a list of strings, not '{v}' : {type(v)}")

    if name.endswith(".__latitudes") or name.endswith(".__longitudes"):
        # This should disappear in the future, when we stop saving a duplicate of lat/lon in the data
        return False

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
        indices = {}
        name_to_index = {}
        for group in self.dataset.keys():
            variables = self.dataset.variables_dict[group]
            ind = np.zeros(len(variables), dtype=bool)
            count = 0
            for j, v in enumerate(variables):
                if self.match_variable(v, group):
                    ind[variables.index(v)] = True
                    indices[group] = ind
                    if group not in name_to_index:
                        name_to_index[group] = {}
                    name_to_index[group][v] = count
                    count += 1
            assert np.sum(ind) == count, f"Mismatch in {group}: {variables}, {ind}"

        self._indices = indices
        self._name_to_index = name_to_index

    def match_variable(self, name, group):
        return match_variable(self._select, name, group)

    def keys(self):
        return self._indices.keys()

    def _load_data(self, i):
        forward = self.dataset._load_data(i)
        data = {}
        for k, v in self._indices.items():
            data[f"latitudes:{k}"] = forward[f"latitudes:{k}"]
            data[f"longitudes:{k}"] = forward[f"longitudes:{k}"]
            data[f"timedeltas:{k}"] = forward[f"timedeltas:{k}"]
            data[f"metadata:{k}"] = forward[f"metadata:{k}"]
        for k, v in self._indices.items():
            data[f"data:{k}"] = forward[f"data:{k}"][v]  # notice the [v] here
        return data

    @property
    def name_to_index(self):
        return self._name_to_index

    @property
    def statistics(self):
        dic = {}
        for group, v in self._indices.items():
            stats = self.dataset.statistics[group]
            dic[group] = {key: stats[key][v] for key in stats.keys()}
            assert "mean" in dic[group], f"Missing mean in {dic[group]}"
        return dic


class Subset(Forward):
    def __init__(self, dataset, indices, reason):
        super().__init__(dataset)

        self.dataset = dataset
        self._indices = indices

        self.reason = reason
        self._dates = dataset.dates[indices]

    @property
    def dates(self):
        return self._dates

    def _load_data(self, i):
        return self.dataset._load_data(self._indices[i])

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
        # todo : update this and write directly the correct nested index in the metadata
        dic = defaultdict(dict)
        for name, i in self.metadata["name_to_index"].items():
            group, k = name.split(".")
            assert isinstance(k, str), f"Invalid name_to_index {name}: {i}"
            assert i[0] == group, f"Invalid name_to_index {name}: {i}"
            assert isinstance(i[1], int), f"Invalid name_to_index {name}: {i}"
            dic[group][k] = i[1]
        return dic

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
        path = os.path.join(self.path, "statistics.npz")
        dic = {}
        for k, v in dict(np.load(path)).items():
            key, group = k.split(":")
            if group not in dic:
                dic[group] = {}
            dic[group][key] = v
        return dic

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
    def _load_data(self, i):
        path = os.path.join(self.path, "data", str(int(i / 10)), f"{i}.npz")
        with open(path, "rb") as f:
            return dict(np.load(f))


class LazyMultiElement(dict):
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __repr__(self):
        d = {group: "<not-loaded>" for group in self.dataset.keys()}
        return str(d)

    def items(self):
        return self._payload.items()

    @property
    def name_to_index(self):
        return self.dataset.name_to_index

    @property
    def name_to_index_dict(self):
        return self.dataset.name_to_index_dict

    @cached_property
    def _payload(self):
        for k in self.dataset._load_data(self.n):
            assert len(k.split(":")) == 2, f"Invalid key {k}"
        return self.dataset._load_data(self.n)

    def keys(self):
        return self.dataset.keys()

    def __getitem__(self, group):
        return self._payload["data:" + group]

    def _get_aux(self, name):
        try:
            return {k: self._payload[name + ":" + k] for k in self.keys()}
        except KeyError as e:
            e.add_note(f"Available keys are {self._payload.keys()}")
            raise

    @property
    def latitudes(self):
        return self._get_aux("latitudes")

    @property
    def longitudes(self):
        return self._get_aux("longitudes")

    @property
    def timedeltas(self):
        return self._get_aux("timedeltas")

    @property
    def statistics(self):
        return self.dataset.statistics


class OneGroup:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def __getitem__(self, i):
        return self.__get(i, "data")

    def __get(self, i, k):
        payload = self.dataset._load_data(i)
        try:
            return payload[k + ":" + self.name]
        except KeyError:
            print(f"KeyError to retrieve {self.name} available groups are", payload.keys())
            raise

    @property
    def statistics(self):
        return self.dataset.statistics[self.name]
