# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import bisect
import datetime
import logging
from functools import cached_property

import numpy as np
import tqdm

from ..buffering import RandomReadBuffer
from ..buffering import WriteBehindBuffer
from . import DateIndexing
from . import date_indexing_registry
from .ranges import DateRange

LOG = logging.getLogger(__name__)


class _Proxy:
    """Proxy class to enable binary search on the first column of a 2D array-like object."""

    def __init__(self, dates) -> None:
        self.dates = dates

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, value: int) -> int:
        return self.dates[value][0]


@date_indexing_registry.register("bisect")
class DateBisect(DateIndexing):
    """Implements date indexing using a bisect (binary search) approach for efficient range queries.

    This class provides methods to bulk load date ranges, retrieve index boundaries, and access
    start and end dates using a Zarr-backed store. It is registered as the 'bisect' date indexing method.
    """

    name = "bisect"

    def __init__(self, /, store: object, **kwargs: object) -> None:
        """Initialise the DateBisect indexer.

        Parameters
        ----------
        store : object
            The backing store (e.g., Zarr group or similar) for the date index ranges.
        **kwargs : object
            Additional keyword arguments (unused).
        """
        self.store = store

    def bulk_load(self, dates_ranges: np.ndarray) -> None:
        """Bulk load date ranges into the backing store, chunked for efficient I/O.

        Parameters
        ----------
        dates_ranges : np.ndarray
            A 2D numpy array of date ranges to be stored, where each row contains epoch, start index, and length.
        """

        assert dates_ranges.dtype == np.int64, dates_ranges.dtype

        row_size = dates_ranges.nbytes // len(dates_ranges)
        chunck_size = 64 * 1024 * 1024 // row_size  # Adjust chunk size to approx 64MB
        LOG.info(f"Bulk loading {dates_ranges.shape} with chunk size {chunck_size}")
        date_index_ranges = self.store.create_dataset(
            "date_index_ranges",
            shape=dates_ranges.shape,
            dtype=dates_ranges.dtype,
            chunks=(chunck_size, 3),
            overwrite=True,
        )
        date_index_ranges.attrs["_ARRAY_DIMENSIONS"] = ["epoch", "start_idx", "length"]

        with WriteBehindBuffer(date_index_ranges) as index:
            total = dates_ranges.shape[0]
            step = chunck_size
            for i in tqdm.tqdm(range(0, total, step)):
                last = min(step, total - i)
                index[i : i + last, :] = dates_ranges[i : i + last, :]

    @cached_property
    def index(self) -> RandomReadBuffer:
        return RandomReadBuffer(self.store["date_index_ranges"])

    def start_end_dates(self) -> tuple[datetime.datetime, datetime.datetime]:
        """Get the start and end dates from the index.

        Returns
        -------
        tuple of datetime.datetime
            The first and last date in the index, as datetime objects.
        """
        first_key, last_key = self.index[0][0], self.index[-1][0]
        return datetime.datetime.fromtimestamp(first_key), datetime.datetime.fromtimestamp(last_key)

    def range_search(self, start: int, end: int, dataset_length: int) -> slice:
        """Find the boundaries in the index for a given start and end epoch.

        This method uses a proxy to perform binary search only on the first dimension of the 2D dates array,
        minimising memory usage by not loading all dates at once.

        start and end are included in the search.

        Parameters
        ----------
        start : int
            The starting epoch (inclusive).
        end : int
            The ending epoch (inclusive).
        dataset_length : int
            The total length of the dataset
        Returns
        -------
        slice
            A slice object representing the range of indices corresponding to the specified start and end epochs.
        """

        assert start < end

        # Search only the first dimension of the 2D dates array, without loading all dates

        start_idx = bisect.bisect_left(_Proxy(self.index), start)
        end_idx = bisect.bisect_left(_Proxy(self.index), end)

        if end_idx == 0 and end < self.index[0][0]:
            # Start edge case: if end is before the first entry
            return slice(0, 0)

        if start_idx == len(self.index):
            # Start edge case: if start is beyond the last entry
            return slice(dataset_length, dataset_length)

        if end_idx == len(self.index):
            end_idx -= 1

        start_entry = DateRange(*self.index[start_idx])
        end_entry = DateRange(*self.index[end_idx])

        if end_entry.epoch > end:
            # There is a GAP in the data
            end_idx -= 1
            end_entry = DateRange(*self.index[end_idx])

        if end_entry.epoch < start_entry.epoch:
            # No data in the range
            return slice(start_entry.offset, start_entry.offset)

        return slice(start_entry.offset, end_entry.offset + end_entry.length)

    def boundaries(self, start: int, end: int) -> tuple[int, int]:

        start_idx = bisect.bisect_left(_Proxy(self.index), start)
        end_idx = bisect.bisect_right(_Proxy(self.index), end)
        return (self.index[start_idx], self.index[end_idx - 1])
