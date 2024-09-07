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
from .forwards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class InterpolateFrequency(Forwards):

    def __init__(self, dataset, frequency):
        super().__init__(dataset)
        self._frequency = frequency_to_timedelta(frequency)

        self.seconds = self._frequency.total_seconds()
        other_seconds = dataset.frequency.total_seconds()

        self.seconds = int(self.seconds)
        assert self.seconds == self._frequency.total_seconds()

        other_seconds = int(other_seconds)
        assert other_seconds == dataset.frequency.total_seconds()

        if self.seconds >= other_seconds:
            raise ValueError(
                f"Interpolate frequency {self._frequency} must be more frequent than dataset frequency {dataset.frequency}"
            )

        if other_seconds % self.seconds != 0:
            raise ValueError(
                f"Interpolate frequency {self._frequency}  must be a multiple of the dataset frequency {dataset.frequency}"
            )

        self.ratio = other_seconds // self.seconds
        self.alphas = np.linspace(0, 1, self.ratio + 1)
        self.other_len = len(dataset)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 0, slice(None))
        result = self._get_slice(previous)
        return apply_index_to_slices_changes(result[index], changes)

    def _get_slice(self, s):
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        if n < 0:
            n += self._len

        if n == self._len - 1:
            # Special case for the last element
            return self.forward[-1]

        i = n // self.ratio
        x = n % self.ratio

        if x == 0:
            # No interpolation needed
            return self.forward[i]

        alpha = self.alphas[x]

        assert 0 < alpha < 1, alpha
        return self.forward[i] * (1 - alpha) + self.forward[i + 1] * alpha

    def __len__(self):
        return (self.other_len - 1) * self.ratio + 1

    @property
    def frequency(self):
        return self._frequency

    @cached_property
    def dates(self):
        result = []
        deltas = [np.timedelta64(self.seconds * i, "s") for i in range(self.ratio)]
        for d in self.forward.dates[:-1]:
            for i in deltas:
                result.append(d + i)
        result.append(self.forward.dates[-1])
        return np.array(result)

    @property
    def shape(self):
        return (self._len,) + self.forward.shape[1:]

    def tree(self):
        return Node(self, [self.forward.tree()], frequency=self.frequency)

    @cached_property
    def missing(self):
        result = []
        j = 0
        for i in range(self.other_len):
            missing = i in self.forward.missing
            for _ in range(self.ratio):
                if missing:
                    result.append(j)
                j += 1

        result = set(x for x in result if x < self._len)
        return result

    def subclass_metadata_specific(self):
        return {
            # "frequency": frequency_to_string(self._frequency),
        }
