# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

####################################################################################################
# All what is below is experimental code that is not tested, it will change completely and should not be used yet
####################################################################################################

import datetime
import logging
from collections import defaultdict
from functools import cached_property

from ..debug import Node
from . import ObservationsBase
from . import ObservationsZarr
from . import make_dates

LOG = logging.getLogger(__name__)
WARNING = "Using experimental code. Unsafe and untested. Use it only if you know what you are doing."


class Multiple(ObservationsBase):
    @property
    def names(self):
        return self._names

    def _check(self):
        names = []
        for ds in self.datasets_as_list():
            for name in ds.variables:
                if name in names:
                    raise ValueError(f"Duplicated variable: {name}. Use rename_prefix to avoid this issue.")

    @classmethod
    def _rearrange_array(cls, data):
        import einops

        if len(data.shape) == 3:
            assert data.shape[1] == 1, f"Expected ensemble dimmension of 1, got {data.shape}"
            data = data[:, 0, :]
        data = einops.rearrange(data, "variables latlon -> latlon variables")
        return data

    @property
    def variables(self):
        variables = []
        for ds in self._datasets_as_list:
            variables += ds.variables
        return variables


def merge_dates(datasets):
    LOG.warning(WARNING)
    start_date = None
    end_date = None
    for d in datasets:
        s, e = min(d.dates), max(d.dates)
        if start_date is None or s < start_date:
            start_date = s
        if end_date is None or e > end_date:
            end_date = e
    return start_date, end_date


class MultipleDict(Multiple):
    def __init__(self, datasets, names=None):
        LOG.warning(WARNING)
        self._names = names
        self.frequency = list(datasets.values())[0].frequency
        for k, d in datasets.items():
            assert d.frequency == self.frequency, f"Expected {self.frequency}, got {d.frequency}"

        start_date, end_date = merge_dates(list(datasets.values()))

        self.datasets = {k: Padded(d, start_date, end_date).mutate() for k, d in datasets.items()}
        self.dates = make_dates(start_date, end_date, self.frequency)

        # todo: implement missing
        assert all(
            d.missing == set() for k, d in self.datasets.items()
        ), f"Expected no missing, got {[d.missing for d in self.datasets]}"
        self.missing = set()

    def tree(self):
        return Node(self, [d.tree() for k, d in self.datasets.items()])
        # return Node(self, {k:d.tree() for k, d in self.datasets.items()})

    @property
    def _datasets_as_list(self):
        return list(self.datasets.values())

    def _datasets_as_dict(self):
        return self.datasets

    def getitem(self, i):
        return {k: self._rearrange_array(d[i]) for k, d in self.datasets.items()}

    @cached_property
    def name_to_index(self):
        dic = {}
        for k, d in self.datasets.items():
            for name in d.variables:
                dic[name] = (k, d.name_to_index[name])
        return dic

    @property
    def statistics(self):
        keys = self._datasets_as_list[0].statistics.keys()
        dic = defaultdict(dict)
        for k, d in self.datasets.items():
            for key in keys:
                dic[key][k] = d.statistics[key]
        assert "mean" in dic, f"Expected 'mean' in statistics, got {list(dic.keys())}"
        return dic

    @property
    def variables_as_dict(self):
        return {k: d.variables for k, d in self.datasets.items()}


class MultipleList(Multiple):
    def __init__(self, datasets, names=None):
        LOG.warning(WARNING)
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
    def _datasets_as_list(self):
        return self.datasets

    def _datasets_as_dict(self):
        return {i: d for i, d in enumerate(self.datasets)}

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])

    def getitem(self, i):
        return [self._rearrange_array(d[i]) for d in self.datasets]

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
        LOG.warning(WARNING)
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
        return self.forward.name_to_index

    @cached_property
    def statistics(self):
        return self.forward.statistics

    @property
    def missing(self):
        return self.forward.missing


class RenamePrefix(Forward):
    def __init__(self, dataset, prefix):
        LOG.warning(WARNING)
        super().__init__(dataset)
        self.prefix = prefix
        self._variables = [f"{prefix}{n}" for n in self.forward.variables]
        for n in self._variables:
            if "-" in n and "_" in n:
                raise ValueError(f"Do nor mix '-' and '_', got {n}")

    @property
    def variables(self):
        return self._variables

    @property
    def name_to_index(self):
        return {k: i for i, k in enumerate(self.variables)}

    def tree(self):
        return Node(self, [self.forward.tree()], rename_prefix=self.prefix)


class Subset(Forward):
    # TODO : delete this class
    def __init__(self, dataset, start, end):
        LOG.warning(WARNING)
        super().__init__(dataset)

        from ..misc import as_first_date
        from ..misc import as_last_date

        self._start = self.forward.dates[0] if start is None else as_first_date(start, self.forward.dates)
        self._end = self.forward.dates[-1] if end is None else as_last_date(end, self.forward.dates)

        self.dates = make_dates(self._start, self._end, self.forward.frequency)

        assert type(self.dates[0]) == type(self.forward.dates[0]), (  # noqa: E721
            type(self.dates[0]),
            type(self.forward.dates[0]),
        )
        _dates = set(self.forward.dates)
        for d in self.dates:
            assert (
                d in _dates
            ), f"Expected {d} in {self.forward.dates[0]}...{self.forward.dates[-1]}, {self._start=}, {self._end=}"
        date_to_index = {date: i for i, date in enumerate(self.forward.dates)}
        self._indices = {i: date_to_index[d] for i, d in enumerate(self.dates)}

        assert len(self._indices) > 0, f"Expected at least one date, got {len(self._indices)}"
        assert len(self.forward.missing) == 0, f"Expected no missing dates, got {self.forward.missing}"

    def getitem(self, i):
        i = self._indices[i]
        return self.forward[i]

    def supporting_arrays(self):
        return dict()

    def tree(self):
        return Node(self, [self.forward.tree()], start=self._start, end=self._end)


class Select(Forward):
    def __init__(self, dataset, select):
        LOG.warning(WARNING)
        super().__init__(dataset)
        self._select = select
        if len(select) != len(set(select)):
            raise ValueError(f"Duplicate variables: {select}")
        assert all(n in self.forward.variables for n in select), f"Expected {select} in {self.forward.variables}"

        self._variables = select
        self._indexes = tuple(self.forward.name_to_index[n] for n in self._variables)

    def tree(self):
        return Node(self, [self.forward.tree()], select=self._select)

    def getitem(self, i):
        data = self.forward[i]
        data = data[self._indexes, :]
        return data

    @property
    def statistics(self):
        dic = {}
        for k, data in self.forward.statistics.items():
            assert len(data.shape) == 1, f"Expected 1D array, got {data.shape}"
            dic[k] = data[self._indexes,]  # notice the "," here because this 1D array is indexed using a tuple.
        return dic

    @property
    def variables(self):
        return self._variables

    @property
    def name_to_index(self):
        return {n: i for i, n in enumerate(self._variables)}


class Padded(Forward):
    def __init__(self, dataset, start, end):
        LOG.warning(WARNING)
        super().__init__(dataset)
        self._frequency = self.forward.frequency
        assert isinstance(self._frequency, datetime.timedelta), f"Expected timedelta, got {type(self._frequency)}"
        self._start_date = start
        self._end_date = end
        self.dates = make_dates(start, end, self._frequency)
        self._indices = {}

        assert type(self.dates[0]) == type(self.forward.dates[0]), (  # noqa: E721
            type(self.dates[0]),
            type(self.forward.dates[0]),
        )

        dates_to_indices = {date: i for i, date in enumerate(self.forward.dates)}
        _forward_dates = set(self.forward.dates)
        assert len(dates_to_indices) == len(_forward_dates)
        self._indices = {j: dates_to_indices[date] for j, date in enumerate(self.dates) if date in _forward_dates}
        assert len(self._indices) == len(_forward_dates), (len(self._indices), len(_forward_dates))

    @property
    def missing(self):
        return set()

    @property
    def frequency(self):
        return self._frequency

    def getitem(self, i):
        # TODO: need to handle dates that are missing in forward
        if i not in self._indices:
            # print(f"❌Requested {i} {self.dates[i]}: No data in {self.forward} ")
            return self.empty_item()
        j = self._indices[i]
        # print(f"  Padding from {i} {self.dates[i]} -> {j} {self.forward.dates[j]}")
        return self.forward[j]

    def tree(self):
        return Node(
            self,
            [self.forward.tree()],
            frequency=self.frequency,
            start=self._start_date,
            end=self._end_date,
        )


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
    if "_start" in kwargs or "_end" in kwargs:
        start = kwargs.pop("_start", None)
        end = kwargs.pop("_end", None)
        dataset = _open_observations(*args, **kwargs).mutate()
        return Subset(dataset, start, end).mutate()

    if "rename_prefix" in kwargs:
        prefix = kwargs.pop("rename_prefix")
        dataset = _open(kwargs).mutate()
        return RenamePrefix(dataset, prefix).mutate()

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
        if isinstance(multiple, list):
            datasets = [_open(d).mutate() for d in multiple]
            return MultipleList(datasets, **kwargs).mutate()
        elif isinstance(multiple, dict):
            datasets = {k: _open(d).mutate() for k, d in multiple.items()}
            return MultipleDict(datasets, **kwargs).mutate()
        raise NotImplementedError(f"Expected list or dict for multiple, got {type(multiple)}")

    if "is_observations" in kwargs:
        kwargs.pop("is_observations")
        assert len(args) == 0, args
        return ObservationsZarr(*args, **kwargs).mutate()

    from ..misc import _open_dataset as _open_fields

    return _open_fields(*args, **kwargs).mutate()
