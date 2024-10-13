# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np

from .debug import Node
from .debug import debug_indexing
from .forwards import Combined
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Merge(Combined):
    def __init__(self, datasets):
        super().__init__(datasets)

        dates = dict()

        for i, d in enumerate(datasets):
            for j, date in enumerate(d.dates):
                date = date.astype(object)
                if date in dates:
                    d1 = datasets[dates[date][0]]
                    d2 = datasets[i]
                    raise ValueError(f"Duplicate date {date} found in datasets {d1} and {d2}")
                dates[date] = (i, j)

        start = min(dates)
        end = max(dates)

        d = datasets[0].frequency
        date = start
        indices = []
        _dates = []
        while date <= end:
            if date not in dates:
                raise ValueError(f"Missing date {date} in dataset {datasets[0]}")
            indices.append(dates[date])
            _dates.append(date)
            date += d

        self._dates = np.array(_dates, dtype="datetime64[s]")
        self._indices = np.array(indices)

    @property
    def dates(self):
        return self._dates

    @cached_property
    def missing(self):
        # TODO: optimize
        result = set()

        for i, (dataset, row) in enumerate(self._indices):
            if row in self.datasets[int(dataset)].missing:
                result.add(i)

        return result

    def check_same_lengths(self, d1, d2):
        # Turned off because we are concatenating along the first axis
        pass

    def check_same_dates(self, d1, d2):
        # Turned off because we are concatenating along the dates axis
        pass

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=0)

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        dataset, row = self._indices[n]
        return self.datasets[int(dataset)][int(row)]


def merge_factory(args, kwargs):

    datasets = kwargs.pop("merge")

    assert isinstance(datasets, (list, tuple))
    assert len(args) == 0

    datasets = [_open(e) for e in datasets]

    if len(datasets) == 1:
        return datasets[0]._subset(**kwargs)

    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Merge(datasets)._subset(**kwargs)
