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
from .debug import Source
from .debug import debug_indexing
from .forewards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import make_slice_or_index_from_list_or_tuple
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


class Subset(Forwards):
    """Select a subset of the dates."""

    def __init__(self, dataset, indices, reason):
        while isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset

        self.dataset = dataset
        self.indices = list(indices)
        self.reason = {k: v for k, v in reason.items() if v is not None}

        # Forward other properties to the super dataset
        super().__init__(dataset)

    @debug_indexing
    def __getitem__(self, n):
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        assert n >= 0, n
        n = self.indices[n]
        return self.dataset[n]

    @debug_indexing
    def _get_slice(self, s):
        # TODO: check if the indices can be simplified to a slice
        # the time checking maybe be longer than the time saved
        # using a slice
        indices = [self.indices[i] for i in range(*s.indices(self._len))]
        indices = make_slice_or_index_from_list_or_tuple(indices)
        if isinstance(indices, slice):
            return self.dataset[indices]
        return np.stack([self.dataset[i] for i in indices])

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, n):
        index, changes = index_to_slices(n, self.shape)
        # print('INDEX', index, changes)
        indices = [self.indices[i] for i in range(*index[0].indices(self._len))]
        indices = make_slice_or_index_from_list_or_tuple(indices)
        # print('INDICES', indices)
        index, _ = update_tuple(index, 0, indices)
        result = self.dataset[index]
        result = apply_index_to_slices_changes(result, changes)
        return result

    def __len__(self):
        return len(self.indices)

    @cached_property
    def shape(self):
        return (len(self),) + self.dataset.shape[1:]

    @cached_property
    def dates(self):
        return self.dataset.dates[self.indices]

    @cached_property
    def frequency(self):
        dates = self.dates
        delta = dates[1].astype(object) - dates[0].astype(object)
        return int(delta.total_seconds() / 3600)

    def source(self, index):
        return Source(self, index, self.forward.source(index))

    def __repr__(self):
        return f"Subset({self.dataset},{self.dates[0]}...{self.dates[-1]}/{self.frequency})"

    @cached_property
    def missing(self):
        return {self.indices[i] for i in self.dataset.missing if i in self.indices}

    def tree(self):
        return Node(self, [self.dataset.tree()], **self.reason)
