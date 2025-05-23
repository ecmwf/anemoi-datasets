# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
from collections import defaultdict
from functools import cached_property

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.data.records.backends import backend_factory

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


def open_records_dataset(dataset, **kwargs):
    if not dataset.endswith(".vz"):
        raise ValueError("dataset must be a .vz file")
    return RecordsDataset(dataset, **kwargs)


class BaseRecordsDataset:

    def __getitem__(self, i):
        if isinstance(i, str):
            return self._getgroup(i)

        if isinstance(i, int):
            return self._getrecord(i)

        raise ValueError(f"Invalid index {i}, must be int or str")

    def _getgroup(self, i):
        return Tabular(self, i)

    def _getrecord(self, i):
        return Record(self, i)

    def _load_data(self, i):
        raise NotImplementedError("Must be implemented in subclass")

    @property
    def start_date(self):
        return self.dates[0]

    @property
    def end_date(self):
        if len(self.dates) == 0:
            return None
        if len(self.dates) == 1:
            return self.dates[0]
        return self.dates[-1]

    @property
    def groups(self):
        return tuple(self.keys())

    def _subset(self, **kwargs):
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        frequency = kwargs.pop("frequency", self.frequency)

        if frequency != self.frequency:
            raise ValueError(f"Changing the frequency {frequency} (from {self.frequency}) is not implemented yet.")

        if start is not None or end is not None:

            def _dates_to_indices(start, end):
                from anemoi.datasets.data.misc import as_first_date
                from anemoi.datasets.data.misc import as_last_date

                start = self.dates[0] if start is None else as_first_date(start, self.dates)
                end = self.dates[-1] if end is None else as_last_date(end, self.dates)

                return [i for i, date in enumerate(self.dates) if start <= date <= end]

            return RecordsSubset(
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
    def name_to_index(self):
        raise NotImplementedError("Must be implemented in subclass")


class RecordsForward(BaseRecordsDataset):
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


def match_variable(lst, group, name):
    # lst must be a list of strings with dots (if there is no dot, it is automatically added at the end)
    # - a dict with keys as group and values as list of strings

    if name == "__latitudes" or name == "__longitudes":
        # This should disappear in the future, when we stop saving a duplicate of lat/lon in the data
        return False

    lst = [k if "." in k else f"{k}.*" for k in lst]

    key = f"{group}.{name}"
    if key in lst:
        return True
    if f"{group}.*" in lst:
        return True
    if f"*.{name}" in lst:
        return True
    if "*" in lst:
        return True
    return False


class Select(RecordsForward):
    def __init__(self, dataset, select):
        super().__init__(dataset)

        self.dataset = dataset

        if isinstance(select, dict):
            # if a dict is provided, make it a list of strings with '.'
            sel = []
            for group, d in select.items():
                for name in d:
                    sel.append(f"{group}.{name}")
            select = sel

        self._select = select

        self.reason = {"select": select}
        self._build_indices_and_name_to_index()

    def _build_indices_and_name_to_index(self):
        indices = {}
        name_to_index = {}
        variables = {}

        # this should be revisited to take into account the order requested by the user
        # see what is done in the fields datasets
        for group, names in self.dataset.variables.items():
            ind = np.zeros(len(names), dtype=bool)
            count = 0
            for j, name in enumerate(names):
                if self.match_variable(group, name):
                    assert j == names.index(name), f"Invalid index {j} for {name} in {group}"
                    ind[j] = True
                    indices[group] = ind
                    if group not in name_to_index:
                        name_to_index[group] = {}
                        assert group not in variables, (group, j, name, variables, name_to_index)
                        variables[group] = []
                    name_to_index[group][name] = count
                    variables[group].append(name)
                    count += 1
            assert np.sum(ind) == count, f"Mismatch in {group}: {names}, {ind}"
        self._indices = indices
        self._name_to_index = name_to_index
        self._variables = variables

    def match_variable(self, *args, **kwargs):
        return match_variable(self._select, *args, **kwargs)

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
    def variables(self):
        return self._variables

    @property
    def statistics(self):
        dic = {}
        for group, v in self._indices.items():
            stats = self.dataset.statistics[group]
            dic[group] = {key: stats[key][v] for key in stats.keys()}
            assert "mean" in dic[group], f"Missing mean in {dic[group]}"
        return dic


class RecordsSubset(RecordsForward):
    def __init__(self, dataset, indices, reason):
        super().__init__(dataset)
        self.dataset = dataset
        self.reason = reason
        self._indices = indices

    @cached_property
    def dates(self):
        return self.dataset.dates[self._indices]

    def _load_data(self, i):
        return self.dataset._load_data(self._indices[i])

    def __len__(self):
        return len(self._indices)


class RecordsDataset(BaseRecordsDataset):

    def __init__(self, path, backend="npz1", **kwargs):
        if kwargs:
            print("Warning: ignoring additional kwargs", kwargs)
        self.path = path
        self.backend = backend_factory(backend, path, **kwargs)
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

    @cached_property
    def metadata(self):
        return self.backend.read_metadata()

    @property
    def shapes(self):
        return self.metadata["shapes"]

    def items(self, *args, **kwargs):
        return {k: Tabular(self, k) for k in self.keys()}.items(*args, **kwargs)

    @cached_property
    def statistics(self):
        return self.backend.read_statistics()

    def __len__(self):
        return len(self.dates)

    @property
    def start_date(self):
        date = self.metadata["start_date"]
        return datetime.datetime.fromisoformat(date)

    @property
    def end_date(self):
        date = self.metadata["end_date"]
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
        return self.backend.read(i)

    def check(self, i=None):
        if i is not None:
            dict_of_sets = defaultdict(set)
            for key in self._load_data(i).keys():
                kind, group = key.split(":")
                dict_of_sets[group].add(kind)
            for group, s in dict_of_sets.items():
                assert s == {"latitudes", "longitudes", "timedeltas", "metadata", "data"}, f"Invalid keys {s}"


class Record(dict):
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

    @cached_property
    def _payload(self):
        payload = self.dataset._load_data(self.n)
        for k in payload.keys():
            assert len(k.split(":")) == 2, f"Invalid key {k}"
        return payload

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

    @property
    def groups(self):
        return tuple(self.keys())


class Tabular:
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    @property
    def group(self):
        return self.name

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
    def variables(self):
        return self.dataset.variables[self.name]

    @property
    def name_to_index(self):
        return self.dataset.name_to_index[self.name]

    @property
    def statistics(self):
        return self.dataset.statistics[self.name]
