# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta

from .debug import Node
from .debug import debug_indexing
from .forwards import Combined
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import length_to_slices
from .indexing import update_tuple
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class ConcatMixin:

    def __len__(self):
        return sum(len(i) for i in self.datasets)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        # print(index, changes)
        lengths = [d.shape[0] for d in self.datasets]
        slices = length_to_slices(index[0], lengths)
        # print("slies", slices)
        result = [d[update_tuple(index, 0, i)[0]] for (d, i) in zip(self.datasets, slices) if i is not None]
        result = np.concatenate(result, axis=0)
        return apply_index_to_slices_changes(result, changes)

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        # TODO: optimize
        k = 0
        while n >= self.datasets[k]._len:
            n -= self.datasets[k]._len
            k += 1
        return self.datasets[k][n]

    @debug_indexing
    def _get_slice(self, s):
        result = []

        lengths = [d.shape[0] for d in self.datasets]
        slices = length_to_slices(s, lengths)

        result = [d[i] for (d, i) in zip(self.datasets, slices) if i is not None]

        return np.concatenate(result)

    @cached_property
    def missing(self):
        result = set()
        offset = 0
        for d in self.datasets:
            result = result | set(m + offset for m in d.missing)
            offset += len(d)
        return result


class Concat(ConcatMixin, Combined):

    def check_compatibility(self, d1, d2):
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=0)

    def check_same_lengths(self, d1, d2):
        # Turned off because we are concatenating along the first axis
        pass

    def check_same_dates(self, d1, d2):
        # Turned off because we are concatenating along the dates axis
        pass

    @property
    def dates(self):
        return np.concatenate([d.dates for d in self.datasets])

    @property
    def shape(self):
        return (len(self),) + self.datasets[0].shape[1:]

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])

    @classmethod
    def check_dataset_compatibility(cls, datasets, fill_missing_gaps=False):
        # Study the dates
        ranges = [(d.dates[0].astype(object), d.dates[-1].astype(object)) for d in datasets]

        # Make sure the dates are disjoint
        for i in range(len(ranges)):
            r = ranges[i]
            for j in range(i + 1, len(ranges)):
                s = ranges[j]
                if r[0] <= s[0] <= r[1] or r[0] <= s[1] <= r[1]:
                    raise ValueError(f"Overlapping dates: {r} and {s} ({datasets[i]} {datasets[j]})")

        # For now we should have the datasets in order with no gaps

        frequency = frequency_to_timedelta(datasets[0].frequency)
        result = []

        for i in range(len(ranges) - 1):
            result.append(datasets[i])
            r = ranges[i]
            s = ranges[i + 1]
            if r[1] + frequency != s[0]:
                if fill_missing_gaps:
                    from .missing import MissingDataset

                    result.append(MissingDataset(datasets[i], r[1] + frequency, s[0] - frequency))
                else:
                    r = [str(e) for e in r]
                    s = [str(e) for e in s]
                    raise ValueError(
                        "Datasets must be sorted by dates, with no gaps: "
                        f"{r} and {s} ({datasets[i]} {datasets[i+1]})"
                    )

        result.append(datasets[-1])
        assert len(result) >= len(datasets), (len(result), len(datasets))

        return result


def concat_factory(args, kwargs):

    datasets = kwargs.pop("concat")
    fill_missing_gaps = kwargs.pop("fill_missing_gaps", False)
    assert isinstance(datasets, (list, tuple))
    assert len(args) == 0

    assert isinstance(datasets, (list, tuple))

    datasets = [_open(e) for e in datasets]

    if len(datasets) == 1:
        return datasets[0]._subset(**kwargs)

    datasets, kwargs = _auto_adjust(datasets, kwargs)

    datasets = Concat.check_dataset_compatibility(datasets, fill_missing_gaps)

    return Concat(datasets)._subset(**kwargs)
