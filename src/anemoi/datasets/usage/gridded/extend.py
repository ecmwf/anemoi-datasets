# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import cached_property
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.dataset import Shape
from anemoi.datasets.usage.dataset import TupleIndex
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.debug import debug_indexing
from anemoi.datasets.usage.forwards import Forwards
from anemoi.datasets.usage.gridded import MissingDateError
from anemoi.datasets.usage.gridded.indexing import apply_index_to_slices_changes
from anemoi.datasets.usage.gridded.indexing import expand_list_indexing
from anemoi.datasets.usage.gridded.indexing import index_to_slices
from anemoi.datasets.usage.gridded.indexing import update_tuple
from anemoi.datasets.usage.misc import as_first_date
from anemoi.datasets.usage.misc import as_last_date

LOG = logging.getLogger(__name__)


class Extend(Forwards):
    """A class to represent a dataset with interpolated frequency."""

    def __init__(self, dataset: Dataset, extend_start: Any = None, extend_end: Any = None) -> None:
        """Initialize the Extend class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be extended.
        extend_start : Any
            The start of the extension.
        extend_end : Any
            The end of the extension.
        """
        super().__init__(dataset)
        self._extend_start = extend_start
        self._extend_end = extend_end

        dates = self.forward.dates
        first = dates[0]
        last = dates[-1]

        frequency = frequency_to_timedelta(self.forward.frequency)
        frequency64 = np.timedelta64(int(frequency.total_seconds()), "s")

        # Unlike `start`/`end` on a subset, we have no reference grid for the
        # extended dates. So we expand a partial date (e.g. a year or a month)
        # to the first / last time step of that period, taking the frequency
        # into account, by passing an empty reference-dates array.
        no_reference = np.array([], dtype=dates.dtype)

        if extend_start is not None:
            extend_start = as_first_date(extend_start, no_reference, frequency=frequency)
            if extend_start > first:
                raise ValueError(f"Start date {extend_start} is after first date {first}")
            self._extend_start = extend_start

        if extend_end is not None:
            extend_end = as_last_date(extend_end, no_reference, frequency=frequency)
            if extend_end < last:
                raise ValueError(f"End date {extend_end} is before last date {last}")
            self._extend_end = extend_end

        if self._extend_start is None:
            self._extend_start = first

        if self._extend_end is None:
            self._extend_end = last

        self._before = (first - self._extend_start) // frequency64
        self._after = (self._extend_end - last) // frequency64

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get the interpolated data for a tuple index.

        Parameters
        ----------
        index : TupleIndex
            The tuple index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The interpolated data for the tuple index.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 0, slice(None))
        result = self._get_slice(previous)
        return apply_index_to_slices_changes(result[index], changes)

    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Get the interpolated data for a slice.

        Parameters
        ----------
        s : slice
            The slice to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The interpolated data for the slice.
        """
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get the interpolated data at the specified index.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve data from.

        Returns
        -------
        NDArray[Any]
            The interpolated data at the specified index.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        if n < 0:
            n += self._len

        if n < self._before:
            raise MissingDateError(f"Date {self.dates[n]} is missing (index={n})")

        if n >= self._before + len(self.forward):
            raise MissingDateError(f"Date {self.dates[n]} is missing (index={n})")

        return self.forward[n - self._before]

    def __len__(self) -> int:
        return self._before + len(self.forward) + self._after

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Get the interpolated dates."""
        result = []
        if self._before > 0:
            result.append(
                np.arange(
                    self._extend_start,
                    self.forward.dates[0],
                    frequency_to_timedelta(self.forward.frequency),
                    dtype="datetime64",
                )
            )
        result.append(self.forward.dates)
        if self._after > 0:
            result.append(
                np.arange(
                    self.forward.dates[-1] + frequency_to_timedelta(self.forward.frequency),
                    self._extend_end + frequency_to_timedelta(self.forward.frequency),
                    frequency_to_timedelta(self.forward.frequency),
                    dtype="datetime64",
                )
            )
        return np.concatenate(result)

    @property
    def shape(self) -> Shape:
        """Get the shape of the interpolated dataset."""
        return (self._len,) + self.forward.shape[1:]

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], extend_start=self._extend_start, extend_end=self._extend_end)

    @cached_property
    def missing(self) -> set[int]:
        """Get the missing data indices."""
        return set(range(self._before)) | self.forward.missing | set(range(len(self.forward) + self._before, self._len))

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the InterpolateFrequency subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the InterpolateFrequency subclass.
        """
        result = {}

        if self._extend_start is not None:
            result["extend_start"] = str(self._extend_start)

        if self._extend_end is not None:
            result["extend_end"] = str(self._extend_end)

        return result
