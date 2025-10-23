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
from collections.abc import Mapping
from functools import cached_property

import numpy as np
from anemoi.utils.config import load_any_dict_format
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.data.debug import Node
from anemoi.datasets.data.records.backends import backend_factory
from anemoi.datasets.data.records.windows import window_from_str

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
    metadata_path = os.path.join(dataset, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    metadata = load_any_dict_format(metadata_path)
    kwargs["backend"] = kwargs.get("backend", metadata["backend"])
    return RecordsDataset(dataset, **kwargs)


def merge_data(list_of_dicts):
    merged = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            merged[key].append(value)
    return {k: np.hstack(v) for k, v in merged.items()}


def _to_numpy_date(d):
    if isinstance(d, np.datetime64):
        assert d.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {d.dtype}"
        return d
    assert isinstance(d, datetime.datetime), f"date must be a datetime.datetime, got {type(d)}"
    return _to_numpy_dates([d])[0]


def _to_numpy_dates(d):
    return np.array(d, dtype="datetime64[s]")


class BaseRecordsDataset:
    """This is the base class for all datasets based on records.
    Records datasets are datasets that can be indexed by time (int) or by group (str).
    A record dataset is designed for observations, where multiple array of difference shapes need to be stored for each date.
    They have the same concept or start_date, end_date, frequency as fields datasets, but each date correspond to a window.
    All windows have the same size (the window span can be different from the dataset frequency)

    variables in a record datasets are identified by a group and a name.
    """

    # Depending on the context, a variable is identified by "group.name",
    # or using a dict with keys as groups and values as list of names.
    # most of the code should be agnostic and transform one format to the other when needed.

    def __getitem__(self, i: int | str):
        if isinstance(i, str):
            return self._getgroup(i)

        if isinstance(i, (int, np.integer)):
            return self._getrecord(i)

        raise ValueError(f"Invalid index {i}, must be int or str")

    @cached_property
    def window(self):
        """Returns a string representation of the relative window of the dataset, such as '(-3h, 3h]'."""
        return str(self._window)

    def _getgroup(self, group: str):
        """Returns a Tabular object for the group. As a partial function when argument group is given but i is not."""
        return Tabular(self, group)

    def _getrecord(self, i: int):
        """Returns a Record object for the time step i. As a partial function when argument i is given but group is not."""
        return Record(self, i)

    def _load_data(self, i: int) -> dict:
        """Load the data for a specific time step or window (i).
        It is expected to return a dict containing keys of the form:

        - "data:group1" : numpy array
        - "latitudes:group1" : numpy array
        - "longitudes:group1" : numpy array
        - "metadata:group1" :
        - ...
        - "data:group2" : numpy array
        - "latitudes:group2" : numpy array
        - ...
        """
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
        raise NotImplementedError("Must be implemented in subclass")

    def _subset(self, **kwargs):
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        frequency = kwargs.pop("frequency", self.frequency)

        if frequency:
            frequency = frequency_to_timedelta(frequency)
            if self.frequency.total_seconds() % frequency.total_seconds() == 0:
                return IncreaseFrequency(self, frequency)
            elif frequency.total_seconds() % self.frequency.total_seconds() == 0:
                raise NotImplementedError("Decreasing frequency not implemented yet")
                # return DecreaseFrequency(self, frequency)
            assert self.frequency == frequency, (self.frequency, frequency)

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

        window = kwargs.pop("window", None)
        if window is not None:
            return Rewindowed(self, window)._subset(**kwargs)

        set_group = kwargs.pop("set_group", None)
        if set_group is not None:
            return SetGroup(self, set_group)._subset(**kwargs)

        rename = kwargs.pop("rename", None)
        if rename is not None:
            return Rename(self, rename)._subset(**kwargs)

        for k in kwargs:
            if k in ["backend"]:
                continue
            raise ValueError(f"Invalid kwargs {kwargs}, must be 'start', 'end', 'frequency' or 'select'")

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

    @property
    def groups(self):
        return self.forward.groups

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
    def _window(self):
        return self.forward._window

    @property
    def shapes(self):
        return self.forward.shapes

    def __len__(self):
        return len(self.dates)

    def tree(self):
        return Node(self, [self.forward.tree()], **self.reason)


class IncreaseFrequency(RecordsForward):
    # change the frequency of a records dataset by splitting the windows to fit the new frequency
    # the new frequency must be a divisor of the original frequency (e.g. 6h -> 3h, but not 3h -> 6h) (and not 6h -> 5h)
    def __init__(self, dataset, frequency):
        super().__init__(dataset)
        self.dataset = dataset
        self._frequency = frequency_to_timedelta(frequency)
        self.reason = {"frequency": frequency}

        self._n = self.dataset.frequency / self._frequency
        if int(self._n) != self._n:
            raise ValueError(f"Cannot split frequency {self.dataset.frequency} to {frequency}, not a multiple")
        self._n = int(self._n)

    @cached_property
    def _window(self):
        previous = self.dataset._window
        if isinstance(previous, int):
            previous = window_from_str(previous)
        return previous / self._n

    def __len__(self):
        return len(self.dataset) * self._n

    @property
    def dates(self):
        dates = []
        for date in self.dataset.dates:
            dates += [date + i * self._frequency for i in range(self._n)]
        return dates

    @property
    def frequency(self):
        return self._frequency

    def metadata(self):
        return self.dataset.metadata

    def _load_data(self, i):
        j = i // self._n
        k = i % self._n

        too_much_data = self.dataset._load_data(j)

        out = {}
        for group in self.groups:
            timedeltas = too_much_data[f"timedeltas:{group}"]
            if timedeltas.dtype != "timedelta64[s]":
                raise ValueError(f"Wrong type for {group}")

            start_delta = self.dataset._window.start + k * self.frequency
            end_delta = start_delta + self._window.end - self._window.start

            def _to_numpy_timedelta(td):
                if isinstance(td, np.timedelta64):
                    assert td.dtype == "timedelta64[s]", f"expecting np.timedelta64[s], got {td.dtype}"
                    return td
                return np.timedelta64(int(td.total_seconds()), "s")

            start_delta = _to_numpy_timedelta(start_delta)
            end_delta = _to_numpy_timedelta(end_delta)
            assert timedeltas.dtype == "timedelta64[s]", f"expecting np.timedelta64[s], got {timedeltas.dtype}"

            if self._window.include_start:
                mask = timedeltas >= start_delta
            else:
                mask = timedeltas > start_delta
            if self._window.include_end:
                mask &= timedeltas <= end_delta
            else:
                mask &= timedeltas < end_delta

            out[f"data:{group}"] = too_much_data[f"data:{group}"][..., mask]
            out[f"latitudes:{group}"] = too_much_data[f"latitudes:{group}"][..., mask]
            out[f"longitudes:{group}"] = too_much_data[f"longitudes:{group}"][..., mask]
            out[f"timedeltas:{group}"] = too_much_data[f"timedeltas:{group}"][..., mask]
            out[f"metadata:{group}"] = too_much_data[f"metadata:{group}"]

        return out

    def tree(self):
        return Node(self, [self.dataset.tree()], **self.reason)


class FieldsRecords(RecordsForward):
    """A wrapper around a FieldsDataset to provide a consistent interface for records datasets."""

    def __init__(self, fields_dataset, name):
        """wrapper around a fields dataset to provide a consistent interface for records datasets.
        A FieldsRecords appears as a RecordsDataset with a single group.
        This allows merging fields datasets with other records datasets.
        Parameters:
             fields_dataset: must be a regular fields dataset
             name: the name of the group
        .
        """
        self.forward = fields_dataset
        from anemoi.datasets.data.dataset import Dataset

        assert isinstance(fields_dataset, Dataset), f"fields_dataset must be a Dataset, got {type(fields_dataset)}"
        self._name = name
        self._groups = [name]
        self.reason = {"name": name}

    @property
    def metadata(self):
        return self.forward.metadata

    def _nest_in_dict(self, obj):
        """Helper to nest the object in a dict with the name as key."""
        return {self._name: obj}

    def _load_data(self, i):
        data = self.forward[i]
        out = {}
        out[f"data:{self._name}"] = data
        out[f"latitudes:{self._name}"] = self.forward.latitudes
        out[f"longitudes:{self._name}"] = self.forward.longitudes
        out[f"timedeltas:{self._name}"] = np.zeros(data.shape[-1], dtype="timedelta64[s]")  # + _to_numpy_date(
        #    self.forward.dates[i]
        # )
        out[f"metadata:{self._name}"] = self.forward.metadata()
        return out

    @property
    def groups(self):
        return self._groups

    @property
    def statistics(self):
        return self._nest_in_dict(self.forward.statistics)

    @property
    def variables(self):
        return self._nest_in_dict(self.forward.variables)

    @property
    def dates(self):
        return self.forward.dates

    @property
    def longitudes(self):
        return self._nest_in_dict(self.forward.longitudes)

    @property
    def latitudes(self):
        return self._nest_in_dict(self.forward.latitudes)

    @property
    def name_to_index(self):
        return self._nest_in_dict(self.forward.name_to_index)

    @property
    def frequency(self):
        return self.forward.frequency

    @property
    def _window(self):
        return self.forward._window

    @property
    def shapes(self):
        return self._nest_in_dict(self.forward.shape)

    def __len__(self):
        return len(self.forward.dates)


class BaseRename(RecordsForward):
    """Renames variables in a records dataset."""

    def __init__(self, dataset, rename):
        self.forward = dataset
        assert isinstance(rename, dict)
        for k, v in rename.items():
            assert isinstance(k, str), k
            assert isinstance(v, str), v
        self.rename = rename
        self.reason = {"rename": rename}

    @property
    def statistics(self):
        return {self.rename.get(k, k): v for k, v in self.forward.statistics.items()}

    @property
    def variables(self):
        return {self.rename.get(k, k): v for k, v in self.forward.variables.items()}

    @property
    def name_to_index(self):
        return {self.rename.get(k, k): v for k, v in self.forward.name_to_index.items()}

    @property
    def groups(self):
        return [self.rename.get(k, k) for k in self.forward.groups]


class Rename(BaseRename):
    pass


class SetGroup(BaseRename):
    def __init__(self, dataset, set_group):
        if len(dataset.groups) != 1:
            raise ValueError(f"{self.__class__.__name__} can only be used with datasets containing a single group.")

        super().__init__(dataset, {dataset.groups[0]: set_group})

    def _load_data(self, i):
        return self.dataset._load_data(i)


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


class Rewindowed(RecordsForward):
    # change the window of a records dataset
    # similar to changing the frequency of a dataset

    def __init__(self, dataset, window):
        super().__init__(dataset)
        self.dataset = dataset

        # in this class anything with 1 refers to the original window/dataset
        # and anything with 2 refers to the new window/dataset

        self._window1 = self.forward._window
        self._window2 = window_from_str(window)
        self.reason = {"window": self.window}

        self._dates1 = _to_numpy_dates(self.forward.dates)
        dates = self._dates1
        self.dates_offset = 0
        while len(dates) > 0 and not self._window1.starts_before(self._dates1, dates, self._window2):
            LOG.warning(f"Removing first date {dates[0]} because it is to early")
            self.dates_offset += 1
            dates = dates[1:]
        while len(dates) > 0 and not self._window1.ends_after(self._dates1, dates, self._window2):
            LOG.warning(f"Removing last date {dates[-1]} because it is to late")
            dates = dates[:-1]

        if len(dates) == 0:
            raise ValueError(
                f"No dates left after rewindowing {self._window1} -> {self._window2} (frequency={self.frequency}), check your window"
            )
        self._dates = dates

        before_span1 = self._window1.start / self.frequency
        before_span2 = self._window2.start / self.frequency
        delta_before_span = before_span2 - before_span1
        if delta_before_span == int(delta_before_span):
            if not self._window1.include_start and self._window2.include_start:
                # if the start of the window is not included, we need to read one more index
                delta_before_span -= 1
        delta_before_span = int(delta_before_span)
        self.delta_before_span = delta_before_span

        after_span1 = self._window1.end / self.frequency
        after_span2 = self._window2.end / self.frequency
        delta_after_span = after_span2 - after_span1
        if delta_after_span == int(delta_after_span):
            if not self._window1.include_end and self._window2.include_end:
                # if the end of the window is not included, we need to read one more index
                delta_after_span += 1
        delta_after_span = int(delta_after_span)
        self.delta_after_span = delta_after_span

    @property
    def window(self):
        return self._window2

    @property
    def dates(self):
        return self._dates

    def __len__(self):
        return len(self.dates)

    @property
    def frequency(self):
        return self.forward.frequency

    def _load_data(self, i):
        print(f"Rewindowing data for i={i} (date={self.dates[i]}) : {self._window1} -> {self._window2}")

        first_j = i + self.delta_before_span
        last_j = i + self.delta_after_span

        first_j = first_j + self.dates_offset
        last_j = last_j + self.dates_offset
        print(f"Requested ds({i}) : need to read {list(range(first_j, last_j + 1))} indices")

        # _load_data could support a list of indices, but for now we merge the data ourselves
        # we merge the windows that we need, and then remove unnecessary data
        too_much_data = merge_data(self.forward._load_data(j) for j in range(first_j, last_j + 1))

        out = {}
        for group in self.groups:
            timedeltas = too_much_data[f"timedeltas:{group}"]
            if timedeltas.dtype != "timedelta64[s]":
                raise ValueError(f"Wrong type for {group}")
            mask = self._window.compute_mask(timedeltas)

            out[f"data:{group}"] = too_much_data[f"data:{group}"][..., mask]
            out[f"latitudes:{group}"] = too_much_data[f"latitudes:{group}"][..., mask]
            out[f"longitudes:{group}"] = too_much_data[f"longitudes:{group}"][..., mask]
            out[f"timedeltas:{group}"] = too_much_data[f"timedeltas:{group}"][..., mask]
            out[f"metadata:{group}"] = too_much_data[f"metadata:{group}"]

        return out


class Select(RecordsForward):
    # Select a subset of variables from a records dataset
    # select can be a list of strings with dots (or a dict with keys as groups and values as list of strings)
    #
    # the selection is a filter, not a reordering, which is different from fields datasets and should be documented/fixed
    #
    # Drop should be implemented

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

    @property
    def metadata(self):
        return dict(select=self._select, forward=self.dataset.metadata)

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
        if not variables:
            raise ValueError(
                f"No variables matched in {self._select} for dataset {self.dataset}. Available groups: {self.dataset.groups} Available variables: {self.dataset.variables} "
            )
        self._indices = indices
        self._name_to_index = name_to_index
        self._variables = variables

    def match_variable(self, *args, **kwargs):
        return match_variable(self._select, *args, **kwargs)

    @property
    def groups(self):
        return list(self._indices.keys())

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
    """Subset of a records dataset based on a list of integer indices."""

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
    """This is the base class for all datasets based on records stored on disk."""

    def __init__(self, path, backend=None, **kwargs):
        if kwargs:
            print("Warning: ignoring additional kwargs", kwargs)
        self.path = path
        self.backend = backend_factory(**backend, path=path)
        self._groups = list(self.metadata["sources"].keys())
        for k in self.groups:
            assert k == self.normalise_key(k), k

    @property
    def groups(self):
        return self._groups

    @classmethod
    def normalise_key(cls, k):
        return "".join([x.lower() if x.isalnum() else "_" for x in k])

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
    def _window(self):
        window = self.metadata["window"]
        return window_from_str(window)

    @cached_property
    def metadata(self):
        return self.backend.read_metadata()

    @property
    def shapes(self):
        return self.metadata["shapes"]

    def items(self, *args, **kwargs):
        return {k: Tabular(self, k) for k in self.groups}.items(*args, **kwargs)

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
        data = self.backend.read(i)
        self.backend._check_data(data)
        return data

    def check(self, i=None):
        if i is not None:
            dict_of_sets = defaultdict(set)
            for key in self._load_data(i).keys():
                kind, group = key.split(":")
                dict_of_sets[group].add(kind)
            for group, s in dict_of_sets.items():
                assert s == {"latitudes", "longitudes", "timedeltas", "metadata", "data"}, f"Invalid keys {s}"

    def tree(self):
        return Node(self, [], path=self.path)


class Record(Mapping):
    """A record corresponds to a single time step in a record dataset."""

    def __init__(self, dataset: RecordsDataset, n: int):
        """A record corresponds to a single time step in a record dataset.
        n : int, the index of the time step in the dataset.
        dataset : RecordsDataset, the dataset this record belongs to.
        """
        self.dataset = dataset
        self.n = n

    def __repr__(self):
        d = {group: "<not-loaded>" for group in self.dataset.groups}
        return str(d)

    def items(self):
        return self._payload.items()

    def __iter__(self):
        return iter(self.groups)

    def __len__(self):
        return len(self.groups)

    def __contains__(self, group):
        return group in self.groups

    @property
    def name_to_index(self):
        return self.dataset.name_to_index

    @cached_property
    def _payload(self) -> dict:
        payload = self.dataset._load_data(self.n)
        for k in payload.keys():
            assert len(k.split(":")) == 2, f"Invalid key {k}"
        return payload

    @cached_property
    def groups(self) -> list[str]:
        return self.dataset.groups

    def __getitem__(self, group):
        k = f"data:{group}"
        if k not in self._payload:
            raise KeyError(f"Group {group} not found in record {self.n}. Available groups are {self.groups}")
        return self._payload[k]

    def _get_aux(self, name):
        try:
            return {k: self._payload[name + ":" + k] for k in self.groups}
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

    def as_dict(self) -> dict:
        """Returns the record as a dictionary with group names as keys.

        Returns
        -------
        dict
            Dictionary mapping group names to their data.
        """
        return {group: self[group] for group in self.groups}


class Tabular:
    """A RecordsDataset for a single group, similar to a fields dataset, but allowing different shapes for each date."""

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
