# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.data.dataset import Dataset
from anemoi.datasets.data.dataset import FullIndex
from anemoi.datasets.data.dataset import Shape
from anemoi.datasets.data.dataset import TupleIndex
from anemoi.datasets.data.debug import Node
from anemoi.datasets.data.debug import debug_indexing
from anemoi.datasets.data.forwards import Forwards
from anemoi.datasets.data.indexing import expand_list_indexing
from anemoi.datasets.data.misc import as_first_date
from anemoi.datasets.data.misc import as_last_date

LOG = logging.getLogger(__name__)


class Padded(Forwards):
    _before: int = 0
    _after: int = 0
    _inside: int = 0

    def __init__(self, dataset: Dataset, start: str, end: str, frequency: str, reason: dict[str, Any]) -> None:
        """Create a padded subset of a dataset.

        Attributes:
        dataset (Dataset): The dataset to subset.
        start (str): The start date of the subset.
        end (str): The end date of the subset.
        frequency (str): The frequency of the subset.
        reason (Dict[str, Any]): The reason for the padding.
        """

        self.reason = {k: v for k, v in reason.items() if v is not None}

        if frequency is None:
            frequency = dataset.frequency

        self._frequency = frequency_to_timedelta(frequency)

        if start is None:
            # default is to start at the first date
            start = dataset.dates[0]
        else:
            start = as_first_date(start, None, frequency=self._frequency)

        if end is None:
            # default is to end at the last date
            end = dataset.dates[-1]
        else:
            end = as_last_date(end, None, frequency=self._frequency)

        assert isinstance(dataset.dates[0], np.datetime64), (dataset.dates[0], type(dataset.dates[0]))

        # 'start' is the requested start date
        # 'end' is the requested end date
        # 'first' is the first date of the dataset
        # 'last' is the last date of the dataset
        first = dataset.dates[0]
        last = dataset.dates[-1]

        timedelta = np.array([frequency], dtype="timedelta64[s]")[0]

        parts = []
        before_end = min(end + timedelta, first)
        before_part = np.arange(start, before_end, timedelta)
        if start < first:
            # if the start date is before the first date of the dataset, there is a "before" part
            assert len(before_part) > 0, (start, first, before_end)
            parts.append(before_part)
            self._before = len(before_part)
        if start >= first:
            # if the start date is the first date of the dataset, there is no "before" part
            assert len(before_part) == 0, (start, first, before_end)
            self._before = 0

        # if the start date is before the last date of the dataset
        # and the end date is after the first date of the dataset
        # there is an "inside" part
        if start < last and end > first:
            inside_start = max(start, first)
            inside_end = min(end, last)
            self.dataset = dataset._subset(start=inside_start, end=inside_end)
            inside_part = self.dataset.dates
            parts.append(inside_part)
            self._inside = len(inside_part)
        else:
            self.dataset = dataset  # still needed to get the empty_item
            self._inside = 0

        after_start = max(start, last + timedelta)
        after_part = np.arange(after_start, end + timedelta, timedelta)
        if end > last:
            # if the end date is after the last date of the dataset, there is an "after" part
            assert len(after_part) > 0, (end, last, after_start)
            parts.append(after_part)
            self._after = len(after_part)
        if end <= last:
            assert len(after_part) == 0, (end, last, after_start)
            self._after = 0

        self._dates = np.hstack(parts)

        assert len(self._dates) == self._before + self._inside + self._after, (
            len(self._dates),
            self._before,
            self._inside,
            self._after,
        )

        assert self._dates[0] == start, (self._dates[0], start)
        assert self._dates[-1] == end, (self._dates[-1], end)

        # Forward other properties to the super dataset
        super().__init__(dataset)

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        if self._i_out_of_range(n):
            return self.empty_item()

        return self.dataset[n - self._before]

    def _i_out_of_range(self, n: FullIndex) -> bool:
        """Check if the index is out of range."""
        if 0 <= n < self._before:
            return True

        if (self._before + self._inside) <= n < (self._before + self._inside + self._after):
            return True
        return False

    @debug_indexing
    def _get_slice(self, s: slice) -> NDArray[Any]:
        LOG.warning("Padded subset does not support slice indexing, returning a list")
        return [self[i] for i in range(*s.indices(self._len))]

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, n: TupleIndex) -> NDArray[Any]:
        LOG.warning("Padded subset does not support tuple indexing, returning a list")
        return [self[i] for i in n]

    def empty_item(self):
        return self.dataset.empty_item()

    def get_aux(self, i: FullIndex) -> NDArray[np.timedelta64]:
        if self._i_out_of_range(i):
            arr = np.array([], dtype=np.float32)
            aux = arr, arr, arr
        else:
            aux = self.dataset.get_aux(i - self._before)

        assert len(aux) == 3, (aux, i)
        return aux

    def __len__(self) -> int:
        return len(self._dates)

    @property
    def frequency(self) -> datetime.timedelta:
        """Get the frequency of the subset."""
        return self._frequency

    @property
    def dates(self) -> NDArray[np.datetime64]:
        return self._dates

    @property
    def shape(self) -> Shape:
        return (len(self.dates),) + self.dataset.shape[1:]

    @cached_property
    def missing(self) -> set[int]:
        raise NotImplementedError("Need to decide whether to include the added dates as missing or not")
        # return self.forward.missing

    def tree(self) -> Node:
        """Get the tree representation of the subset.

        Returns:
        Node: The tree representation of the subset.
        """
        return Node(self, [self.dataset.tree()], **self.reason)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns:
        Dict[str, Any]: The metadata specific to the forwards subclass.
        """
        return {
            # "indices": self.indices,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        """Get the string representation of the subset.

        Returns:
        str: The string representation of the subset.
        """
        return f"Padded({self.forward}, {self.dates[0]}...{self.dates[-1]}, frequency={self.frequency})"
