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
from typing import Union

import numpy as np
from numpy.typing import NDArray

from anemoi.datasets.create.utils import to_datetime
from anemoi.datasets.data import MissingDateError

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import TupleIndex
from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import expand_list_indexing
from .indexing import update_tuple

LOG = logging.getLogger(__name__)

# TODO: Use that class instead of ZarrMissing


class MissingDates(Forwards):
    """Handles missing dates in a dataset.

    Attributes
    ----------
    dataset : Dataset
        The dataset object.
    missing_dates : List[Union[int, str]]
        List of missing dates.
    """

    def __init__(self, dataset: Dataset, missing_dates: List[Union[int, str]]) -> None:
        """Initializes the MissingDates class.

        Parameters
        ----------
        dataset : Dataset
            The dataset object.
        missing_dates : List[Union[int, str]]
            List of missing dates.
        """
        super().__init__(dataset)
        self.missing_dates = []

        self._missing = set()

        other = []
        for date in missing_dates:
            if isinstance(date, int):
                self._missing.add(date)
                self.missing_dates.append(dataset.dates[date])
            else:
                date = to_datetime(date)
                other.append(date)

        if other:
            for i, date in enumerate(dataset.dates):
                if date in other:
                    self._missing.add(i)
                    self.missing_dates.append(date)

        n = self.forward._len
        self._missing = set(i for i in self._missing if 0 <= i < n)
        self.missing_dates = sorted(to_datetime(x) for x in self.missing_dates)

        assert len(self._missing), "No dates to force missing"

    @cached_property
    def missing(self) -> Set[int]:
        """Returns the set of missing indices."""
        return self._missing.union(self.forward.missing)

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Retrieves the item at the given index.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve.

        Returns
        -------
        NDArray[Any]
            The item at the given index.
        """
        if isinstance(n, int):
            if n in self.missing:
                self._report_missing(n)
            return self.forward[n]

        if isinstance(n, slice):
            common = set(range(*n.indices(len(self)))) & self.missing
            if common:
                self._report_missing(list(common)[0])
            return self.forward[n]

        if isinstance(n, tuple):
            first = n[0]
            if isinstance(first, int):
                if first in self.missing:
                    self._report_missing(first)
                return self.forward[n]

            if isinstance(first, slice):
                common = set(range(*first.indices(len(self)))) & self.missing
                if common:
                    self._report_missing(list(common)[0])
                return self.forward[n]

            if isinstance(first, (list, tuple)):
                common = set(first) & self.missing
                if common:
                    self._report_missing(list(common)[0])
                return self.forward[n]

        raise TypeError(f"Unsupported index {n} {type(n)}")

    def _report_missing(self, n: int) -> None:
        """Reports a missing date.

        Parameters
        ----------
        n : int
            The index of the missing date.
        """
        raise MissingDateError(f"Date {self.forward.dates[n]} is missing (index={n})")

    @property
    def reason(self) -> Dict[str, Any]:
        """Provides the reason for missing dates."""
        return {"missing_dates": self.missing_dates}

    def tree(self) -> Node:
        """Builds a tree representation of the missing dates.

        Returns
        -------
        Node
            The tree representation of the missing dates.
        """
        return Node(self, [self.forward.tree()], **self.reason)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Provides metadata specific to the subclass.

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the subclass.
        """
        return {"missing_dates": self.missing_dates}


class SkipMissingDates(Forwards):
    """Skips missing dates in a dataset.

    Attributes
    ----------
    dataset : Dataset
        The dataset object.
    expected_access : Union[int, slice]
        The expected access pattern.
    """

    def __init__(self, dataset: Dataset, expected_access: Union[int, slice]) -> None:
        """Initializes the SkipMissingDates class.

        Parameters
        ----------
        dataset : Dataset
            The dataset object.
        expected_access : Union[int, slice]
            The expected access pattern.
        """
        super().__init__(dataset)

        # if isinstance(expected_access, (tuple, list)):
        #     expected_access = slice(*expected_access)

        if isinstance(expected_access, int):
            expected_access = slice(0, expected_access)

        assert isinstance(expected_access, slice), f"Expected access must be a slice, got {expected_access}"

        expected_access = slice(*expected_access.indices(dataset._len))
        missing = dataset.missing.copy()

        size = (expected_access.stop - expected_access.start) // expected_access.step
        indices = []

        for i in range(dataset._len):
            s = slice(expected_access.start + i, expected_access.stop + i, expected_access.step)
            p = set(range(*s.indices(dataset._len)))
            if p.intersection(missing):
                continue

            if len(p) != size:
                continue

            indices.append(tuple(sorted(p)))

        self.expected_access = expected_access
        self.indices = indices

    def __len__(self) -> int:
        """Returns the length of the indices.

        Returns
        -------
        int
            The length of the indices.
        """
        return len(self.indices)

    @property
    def start_date(self) -> np.datetime64:
        """Returns the start date."""
        return self.forward.start_date

    @property
    def end_date(self) -> np.datetime64:
        """Returns the end date."""
        return self.forward.end_date

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Not implemented. Raises an error."""
        raise NotImplementedError("SkipMissingDates.dates")

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Retrieves a tuple of items at the given index.

        Parameters
        ----------
        index : TupleIndex
            The index to retrieve.

        Returns
        -------
        NDArray[Any]
            The tuple of items at the given index.
        """

        def _get_one(n):
            result = []
            for i in self.indices[n]:
                s, _ = update_tuple(index, 0, i)
                result.append(self.forward[s])

            return tuple(result)

        first = index[0]
        if isinstance(first, int):
            return _get_one(first)

        assert isinstance(first, slice), f"SkipMissingDates._get_tuple {index}"

        values = [_get_one(i) for i in range(*first.indices(self._len))]

        result = [_ for _ in zip(*values)]
        return tuple(np.stack(_) for _ in result)

    @debug_indexing
    def _get_slice(self, s: slice) -> Tuple[NDArray[Any], ...]:
        """Retrieves a slice of items.

        Parameters
        ----------
        s : slice
            The slice to retrieve.

        Returns
        -------
        Tuple[NDArray[Any], ...]
            The slice of items.
        """
        values = [self[i] for i in range(*s.indices(self._len))]
        result = [_ for _ in zip(*values)]
        return tuple(np.stack(_) for _ in result)

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> Tuple[NDArray[Any], ...]:
        """Retrieves the item at the given index.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve.

        Returns
        -------
        Tuple[NDArray[Any], ...]
            The item at the given index.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        return tuple(self.forward[i] for i in self.indices[n])

    @property
    def frequency(self) -> datetime.timedelta:
        """Returns the frequency of the dataset."""
        return self.forward.frequency

    def tree(self) -> Node:
        """Builds a tree representation of the skipped missing dates.

        Returns
        -------
        Node
            The tree representation of the skipped missing dates.
        """
        return Node(self, [self.forward.tree()], expected_access=self.expected_access)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Provides metadata specific to the subclass.

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the subclass.
        """
        return {"expected_access": self.expected_access}


class MissingDataset(Forwards):
    """Represents a dataset with missing dates.

    Attributes
    ----------
    dataset : Dataset
        The dataset object.
    start : np.datetime64
        The start date.
    end : np.datetime64
        The end date.
    """

    def __init__(self, dataset: Dataset, start: np.datetime64, end: np.datetime64) -> None:
        """Initializes the MissingDataset class.

        Parameters
        ----------
        dataset : Dataset
            The dataset object.
        start : np.datetime64
            The start date.
        end : np.datetime64
            The end date.
        """
        super().__init__(dataset)
        self.start = start
        self.end = end

        dates = []
        date = start
        while date <= end:
            dates.append(date)
            date += dataset.frequency

        self._dates = np.array(dates, dtype="datetime64")
        self._missing = set(range(len(dates)))

    def __len__(self) -> int:
        """Returns the length of the dates.

        Returns
        -------
        int
            The length of the dates.
        """
        return len(self._dates)

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Returns the dates of the dataset."""
        return self._dates

    @property
    def missing(self) -> Set[int]:
        """Returns the set of missing indices."""
        return self._missing

    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Raises an error for missing dates.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve.

        Raises
        ------
        MissingDateError
            If the date is missing.

        Returns:
            NDArray[Any]: The data at the specified index.
        """
        raise MissingDateError(f"Date {self.dates[n]} is missing (index={n})")

    def tree(self) -> Node:
        """Builds a tree representation of the missing dataset.

        Returns
        -------
        Node
            The tree representation of the missing dataset.
        """
        return Node(self, [self.forward.tree()], start=self.start, end=self.end)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Provides metadata specific to the subclass.

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the subclass.
        """
        return {"start": self.start, "end": self.end}
