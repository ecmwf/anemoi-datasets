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
from typing import Dict
from typing import Set

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.data.dataset import FullIndex
from anemoi.datasets.data.dataset import Shape
from anemoi.datasets.data.dataset import TupleIndex
from anemoi.datasets.data.debug import Node
from anemoi.datasets.data.debug import debug_indexing
from anemoi.datasets.data.forwards import Forwards
from anemoi.datasets.data.indexing import expand_list_indexing

LOG = logging.getLogger(__name__)


def _normalise_date(date, default):
    if date is None:
        date = default

    if isinstance(date, str):
        try:
            date = datetime.datetime.fromisoformat(date)
        except ValueError:
            raise ValueError(f"Invalid date {date}, only isoformat is supported with padding")

    if isinstance(date, datetime.datetime):
        date = np.datetime64(date, "s")

    assert isinstance(date, np.datetime64), (date, type(date))

    return date


class Padded(Forwards):
    _before: int = 0
    _after: int = 0
    _inside: int = 0

    def __init__(self, dataset, start, end, frequency, reason):
        self.reason = {k: v for k, v in reason.items() if v is not None}

        if frequency is None:
            frequency = dataset.frequency

        self._frequency = frequency_to_timedelta(frequency)

        start = _normalise_date(start, dataset.dates[0])
        end = _normalise_date(end, dataset.dates[-1])

        assert isinstance(dataset.dates[0], np.datetime64), (dataset.dates[0], type(dataset.dates[0]))

        timedelta = np.array([frequency], dtype="timedelta64[s]")[0]

        dates_parts = []

        if start < dataset.dates[0]:
            dates_parts.append(np.arange(start, dataset.dates[0], timedelta))
            self._before = len(dates_parts[-1])

        dates_parts.append(dataset.dates)
        self._inside = len(dates_parts[-1])

        if end > dataset.dates[-1]:
            dates_parts.append(np.arange(dataset.dates[-1] + timedelta, end + timedelta, timedelta))
            self._after = len(dates_parts[-1])

        self._dates = np.hstack(dates_parts)
        assert len(self._dates) == self._before + self._inside + self._after, (
            len(self._dates),
            self._before,
            self._inside,
            self._after,
        )

        # Forward other properties to the super dataset
        super().__init__(dataset)

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        if 0 <= n < self._before:
            return self.empty_item()

        if (self._before + self._inside) <= n < (self._before + self._inside + self._after):
            return self.empty_item()

        return self.forward[n - self._before]

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
        return self.forward.empty_item()

    def __len__(self) -> int:
        print("len", len(self._dates))
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
        return (len(self.dates),) + self.forward.shape[1:]

    @cached_property
    def missing(self) -> Set[int]:
        raise NotImplementedError("Need to decide whether to include the added dates as missing or not")
        # return self.forward.missing

    def tree(self) -> Node:
        """Get the tree representation of the subset.

        Returns:
        Node: The tree representation of the subset.
        """
        return Node(self, [self.forward.tree()], **self.reason)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns:
        Dict[str, Any]: The metadata specific to the forwards subclass.
        """
        return {
            # "indices": self.indices,
            "reason": self.reason,
        }
