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
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.data.debug import Node
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
    return RecordsDataset(dataset, **kwargs)


def merge_data(list_of_dicts):
    merged = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            merged[key].append(value)
    return {k: np.hstack(v) for k, v in merged.items()}


def _to_numpy_timedelta(td):
    if isinstance(td, np.timedelta64):
        assert td.dtype == "timedelta64[s]", f"expecting np.timedelta64[s], got {td.dtype}"
        return td
    return np.timedelta64(int(td.total_seconds()), "s")


def _to_numpy_date(d):
    if isinstance(d, np.datetime64):
        assert d.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {d.dtype}"
        return d
    assert isinstance(d, datetime.datetime), f"date must be a datetime.datetime, got {type(d)}"
    return _to_numpy_dates([d])[0]


def _to_numpy_dates(d):
    return np.array(d, dtype="datetime64[s]")


class BaseRecordsDataset:

    def __getitem__(self, i):
        if isinstance(i, str):
            return self._getgroup(i)

        if isinstance(i, int):
            return self._getrecord(i)

        raise ValueError(f"Invalid index {i}, must be int or str")

    @cached_property
    def window(self):
        return str(self._window)

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
        raise NotImplementedError("Must be implemented in subclass")

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


class FieldsRecords(RecordsForward):
    """A wrapper around a FieldsDataset to provide a consistent interface for records datasets."""

    def __init__(self, fields_dataset, name):
        self.forward = fields_dataset
        self._name = name
        self._groups = [name]
        self.reason = {"name": name}

    def _nest_in_dict(self, obj):
        """Helper to nest the object in a dict with the name as key."""
        return {self._name: obj}

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


class GenericRename(RecordsForward):
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


class Rename(GenericRename):
    pass


class SetGroup(GenericRename):
    def __init__(self, dataset, set_group):
        if len(dataset.groups) != 1:
            raise ValueError(f"{self.__class__.__name__} can only be used with datasets containing a single group.")

        super.__init__(dataset, {dataset.groups[0]: set_group})


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


def window_from_str(txt):
    """Parses a window string of the form '(-6h, 0h]' and returns a WindowsSpec object."""
    if txt.startswith("["):
        include_start = True
    elif txt.startswith("("):
        include_start = False
    else:
        raise ValueError(f"Invalid window {txt}, must start with '(' or '['")
    txt = txt[1:]

    if txt.endswith("]"):
        include_end = True
    elif txt.endswith(")"):
        include_end = False
    else:
        raise ValueError(f"Invalid window {txt}, must end with ')' or ']'")
    txt = txt[:-1]

    txt = txt.strip()
    if ";" in txt:
        txt = txt.replace(";", ",")
    lst = txt.split(",")
    if len(lst) != 2:
        raise ValueError(
            f"Invalid window {txt}, must be of the form '(start, end)' or '[start, end]' or '[start, end)' or '(start, end]'"
        )
    start, end = lst
    start = start.strip()
    end = end.strip()

    def _to_timedelta(t):
        # This part should go into utils
        from anemoi.utils.dates import as_timedelta

        if t.startswith(" ") or t.endswith(" "):
            t = t.strip()
        if t.startswith("-"):
            return -as_timedelta(t[1:])
        if t.startswith("+"):
            return as_timedelta(t[1:])
        # end of : This part should go into utils
        return as_timedelta(t)

    start = _to_timedelta(start)
    end = _to_timedelta(end)
    return WindowsSpec(
        start=start,
        end=end,
        include_start=include_start,
        include_end=include_end,
    )


class AbsoluteWindow:
    def __init__(self, start, end, include_start=True, include_end=True):
        assert isinstance(start, datetime.datetime), f"start must be a datetime.datetime, got {type(start)}"
        assert isinstance(end, datetime.datetime), f"end must be a datetime.datetime, got {type(end)}"
        assert isinstance(include_start, bool), f"include_start must be a bool, got {type(include_start)}"
        assert isinstance(include_end, bool), f"include_end must be a bool, got {type(include_end)}"
        if start >= end:
            raise ValueError(f"start {start} must be less than end {end}")
        self.start = start
        self.end = end
        self.include_start = include_start
        self.include_end = include_end

    def __repr__(self):
        return f"{'[' if self.include_start else '('}{self.start.isoformat()},{self.end.isoformat()}{']' if self.include_end else ')'}"


class WindowsSpec:
    def __init__(self, *, start, end, include_start=False, include_end=True):
        assert isinstance(start, (str, datetime.timedelta)), f"start must be a str or timedelta, got {type(start)}"
        assert isinstance(end, (str, datetime.timedelta)), f"end must be a str or timedelta, got {type(end)}"
        assert isinstance(include_start, bool), f"include_start must be a bool, got {type(include_start)}"
        assert isinstance(include_end, bool), f"include_end must be a bool, got {type(include_end)}"
        assert include_start in (True, False), f"Invalid include_start {include_start}"  # None is not allowed
        assert include_end in (True, False), f"Invalid include_end {include_end}"  # None is not allowed
        if start >= end:
            raise ValueError(f"start {start} must be less than end {end}")
        self.start = start
        self.end = end
        self.include_start = include_start
        self.include_end = include_end

        self._start_np = _to_numpy_timedelta(start)
        self._end_np = _to_numpy_timedelta(end)

    def to_absolute_window(self, date):
        """Convert the window to an absolute window based on a date."""
        assert isinstance(date, datetime.datetime), f"date must be a datetime.datetime, got {type(date)}"
        start = date + self.start
        end = date + self.end
        return AbsoluteWindow(start=start, end=end, include_start=self.include_start, include_end=self.include_end)

    def __repr__(self):
        first = "[" if self.include_start else "("
        last = "]" if self.include_end else ")"

        def _frequency_to_string(t):
            if t < datetime.timedelta(0):
                return f"-{frequency_to_string(-t)}"
            elif t == datetime.timedelta(0):
                return "0"
            return frequency_to_string(t)

        return f"{first}{_frequency_to_string(self.start)},{_frequency_to_string(self.end)}{last}"

    def compute_mask(self, timedeltas):
        assert timedeltas.dtype == "timedelta64[s]", f"expecting np.timedelta64[s], got {timedeltas.dtype}"
        if self.include_start:
            lower_mask = timedeltas >= self._start_np
        else:
            lower_mask = timedeltas > self._start_np

        if self.include_end:
            upper_mask = timedeltas <= self._end_np
        else:
            upper_mask = timedeltas < self._end_np

        return lower_mask & upper_mask

    def starts_before(self, my_dates, other_dates, other_window):
        assert my_dates.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {my_dates.dtype}"
        assert other_dates.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {other_dates.dtype}"
        assert isinstance(other_window, WindowsSpec), f"other_window must be a WindowsSpec, got {type(other_window)}"

        my_start = my_dates[0] + self._start_np
        other_start = other_dates[0] + other_window._start_np

        if my_start == other_start:
            return (not other_window.include_start) or self.include_start
        return my_start <= other_start

    def ends_after(self, my_dates, other_dates, other_window):
        assert my_dates.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {my_dates.dtype}"
        assert other_dates.dtype == "datetime64[s]", f"expecting np.datetime64[s], got {other_dates.dtype}"
        assert isinstance(other_window, WindowsSpec), f"other_window must be a WindowsSpec, got {type(other_window)}"

        my_end = my_dates[-1] + self._end_np
        other_end = other_dates[-1] + other_window._end_np

        if my_end == other_end:
            print(".", (not other_window.include_end) or self.include_end)
            return (not other_window.include_end) or self.include_end
        print(my_end >= other_end)
        return my_end >= other_end


class Rewindowed(RecordsForward):
    def __init__(self, dataset, window):
        super().__init__(dataset)
        self.dataset = dataset

        # in this class anything with 1 refers to the original window/dataset
        # and anything with 2 refers to the new window/dataset
        # and we use _Δ for timedeltas

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
        return "".join([x.lower() if x.isalnum() else "-" for x in k])

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


class Record:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __repr__(self):
        d = {group: "<not-loaded>" for group in self.dataset.groups}
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

    @cached_property
    def groups(self):
        return self.dataset.groups

    def __getitem__(self, group):
        return self._payload["data:" + group]

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
