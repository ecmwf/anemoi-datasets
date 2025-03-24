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
from typing import List
from typing import Set
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from . import MissingDateError
from .dataset import Dataset
from .dataset import FullIndex
from .dataset import TupleIndex
from .debug import Node
from .debug import debug_indexing
from .forwards import Combined
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Merge(Combined):
    """A class to merge multiple datasets along the dates axis, handling gaps in dates if allowed."""

    def __init__(self, datasets: List[Dataset], allow_gaps_in_dates: bool = False) -> None:
        """Initialize the Merge object.

        Parameters
        ----------
        datasets : List[Dataset]
            List of datasets to merge.
        allow_gaps_in_dates : bool, optional
            Whether to allow gaps in dates. Defaults to False.
        """
        super().__init__(datasets)

        self.allow_gaps_in_dates = allow_gaps_in_dates

        dates = dict()  # date -> (dataset_index, date_index)

        for i, d in enumerate(datasets):
            for j, date in enumerate(d.dates):
                date = date.astype(object)
                if date in dates:

                    d1 = datasets[dates[date][0]]  # Selected
                    d2 = datasets[i]  # The new one

                    if j in d2.missing:
                        # LOG.warning(f"Duplicate date {date} found in datasets {d1} and {d2}, but {date} is missing in {d}, ignoring")
                        continue

                    k = dates[date][1]
                    if k in d1.missing:
                        # LOG.warning(f"Duplicate date {date} found in datasets {d1} and {d2}, but {date} is missing in {d}, ignoring")
                        dates[date] = (i, j)  # Replace the missing date with the new one
                        continue

                    raise ValueError(f"Duplicate date {date} found in datasets {d1} and {d2}")
                else:
                    dates[date] = (i, j)

        all_dates = sorted(dates)
        start = all_dates[0]
        end = all_dates[-1]

        frequency = min(d2 - d1 for d1, d2 in zip(all_dates[:-1], all_dates[1:]))

        date = start
        indices = []
        _dates = []

        self._missing_index = len(datasets)

        while date <= end:
            if date not in dates:
                if self.allow_gaps_in_dates:
                    dates[date] = (self._missing_index, -1)
                else:
                    raise ValueError(
                        f"merge: date {date} not covered by dataset. Start={start}, end={end}, frequency={frequency}"
                    )

            indices.append(dates[date])
            _dates.append(date)
            date += frequency

        self._dates = np.array(_dates, dtype="datetime64[s]")
        self._indices = np.array(indices)
        self._frequency = frequency.astype(object)

    def __len__(self) -> int:
        """Get the number of dates in the merged dataset.

        Returns
        -------
        int
            Number of dates.
        """
        return len(self._dates)

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Get the dates of the merged dataset."""
        return self._dates

    @property
    def frequency(self) -> datetime.timedelta:
        """Get the frequency of the dates in the merged dataset."""
        return self._frequency

    @cached_property
    def missing(self) -> Set[int]:
        """Get the indices of missing dates in the merged dataset."""
        # TODO: optimize
        result: Set[int] = set()

        for i, (dataset, row) in enumerate(self._indices):
            if dataset == self._missing_index:
                result.add(i)
                continue

            if row in self.datasets[dataset].missing:
                result.add(i)

        return result

    def check_same_lengths(self, d1: Dataset, d2: Dataset) -> None:
        """Check if the lengths of two datasets are the same. (Disabled for merging).

        Parameters
        ----------
        d1 : Dataset
            First dataset.
        d2 : Dataset
            Second dataset.
        """
        # Turned off because we are concatenating along the first axis
        pass

    def check_same_dates(self, d1: Dataset, d2: Dataset) -> None:
        """Check if the dates of two datasets are the same. (Disabled for merging).

        Parameters
        ----------
        d1 : Dataset
            First dataset.
        d2 : Dataset
            Second dataset.
        """
        # Turned off because we are concatenating along the dates axis
        pass

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Check if two datasets are compatible for merging.

        Parameters
        ----------
        d1 : Dataset
            First dataset.
        d2 : Dataset
            Second dataset.
        """
        super().check_compatibility(d1, d2)
        self.check_same_sub_shapes(d1, d2, drop_axis=0)

    def tree(self) -> Node:
        """Get the tree representation of the merged dataset.

        Returns
        -------
        Node
            Tree representation of the merged dataset.
        """
        return Node(self, [d.tree() for d in self.datasets], allow_gaps_in_dates=self.allow_gaps_in_dates)

    def metadata_specific(self) -> Dict[str, Any]:
        """Get the specific metadata for the merged dataset.

        Returns
        -------
        Dict[str, Any]
            Specific metadata.
        """
        return {"allow_gaps_in_dates": self.allow_gaps_in_dates}

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get the item at the specified index.

        Parameters
        ----------
        n : FullIndex
            Index to retrieve.

        Returns
        -------
        NDArray[Any]
            Retrieved item.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        dataset, row = self._indices[n]

        if dataset == self._missing_index:
            raise MissingDateError(f"Date {self.dates[n]} is missing (index={n})")

        return self.datasets[dataset][int(row)]

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get the item at the specified tuple index.

        Parameters
        ----------
        index : TupleIndex
            Tuple index to retrieve.

        Returns
        -------
        NDArray[Any]
            Retrieved item.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 0, slice(None))
        result = self._get_slice(previous)
        return apply_index_to_slices_changes(result[index], changes)

    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Get the items in the specified slice.

        Parameters
        ----------
        s : slice
            Slice to retrieve.

        Returns
        -------
        NDArray[Any]
            Retrieved items.
        """
        return np.stack([self[i] for i in range(*s.indices(self._len))])


def merge_factory(args: Tuple, kwargs: Dict[str, Any]) -> Dataset:
    """Factory function to create a merged dataset.

    Parameters
    ----------
    args : Tuple
        Positional arguments.
    kwargs : Dict[str, Any]
        Keyword arguments.

    Returns
    -------
    Dataset
        Merged dataset.
    """
    datasets = kwargs.pop("merge")

    assert isinstance(datasets, (list, tuple))
    assert len(args) == 0

    datasets = [_open(e) for e in datasets]

    if len(datasets) == 1:
        return datasets[0]._subset(**kwargs)

    datasets, kwargs = _auto_adjust(datasets, kwargs)

    allow_gaps_in_dates = kwargs.pop("allow_gaps_in_dates", False)

    return Merge(datasets, allow_gaps_in_dates=allow_gaps_in_dates)._subset(**kwargs)
