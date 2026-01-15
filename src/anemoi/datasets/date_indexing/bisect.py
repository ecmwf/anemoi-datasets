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

from ..buffering import ReadAheadWriteBehindBuffer
from ..debug import extract_dates_from_results as _
from . import DateIndexing
from . import date_indexing_registry
from .ranges import DateRange

LOG = logging.getLogger(__name__)


class _Proxy:
    """Proxy class to enable binary search on the first column of a 2D array-like object."""

    def __init__(self, dates: ReadAheadWriteBehindBuffer) -> None:
        """Initialise the Proxy.

        Parameters
        ----------
        dates : ReadAheadWriteBehindBuffer
            The ReadAheadWriteBehindBuffer object containing date index data.
        """
        self.dates = dates

    def __len__(self) -> int:
        """Return the number of rows in the dates array."""
        return len(self.dates)

    def __getitem__(self, value: int) -> int:
        """Get the epoch value from the first column of the specified row.

        Parameters
        ----------
        value : int
            Row index.

        Returns
        -------
        int
            The epoch value at the specified row.
        """
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

        with ReadAheadWriteBehindBuffer(date_index_ranges) as index:
            total = dates_ranges.shape[0]
            step = chunck_size
            for i in tqdm.tqdm(range(0, total, step)):
                last = min(step, total - i)
                index[i : i + last, :] = dates_ranges[i : i + last, :]

    @cached_property
    def index(self) -> ReadAheadWriteBehindBuffer:
        return ReadAheadWriteBehindBuffer(self.store["date_index_ranges"])

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
        adjust_end = False

        # Search only the first dimension of the 2D dates array, without loading all dates
        start_idx = bisect.bisect_left(_Proxy(self.index), start)
        if start_idx == len(self.index):
            # End edge case: if start is beyond the last entry
            return slice(dataset_length, dataset_length)

        start_entry = DateRange(*self.index[start_idx])

        end_idx = bisect.bisect_left(_Proxy(self.index), end)
        # End edge case: if end is beyond the last entry, adjust end_idx
        if end_idx == len(self.index):
            adjust_end = True
            end_idx -= 1
        # assert 0 <= end_idx < len(self.index), (end_idx, end, len(self.index))
        end_entry = DateRange(*self.index[end_idx])

        assert start_idx <= end_idx, (start_idx, end_idx, start, end)
        assert dataset_length is None or end_entry.offset + end_entry.length <= dataset_length, (
            end_entry.offset,
            end_entry.length,
            dataset_length,
        )
        print("Searching for range:", _(start), _(end))
        print("Start/end entries:", start_entry, end_entry)

        diff_s = (int(start_entry.epoch) > int(start)) - (int(start_entry.epoch) < int(start))
        diff_e = (int(end_entry.epoch) > int(end)) - (int(end_entry.epoch) < int(end))

        match (diff_s, diff_e):
            case (0, 0):
                # Both entries match exactly what was searched
                return slice(start_entry.offset, end_entry.offset + end_entry.length)

            case (1, 0):
                # Start entry is after the searched start, end entry matches exactly
                return slice(start_entry.offset, end_entry.offset + end_entry.length)

            case (0, 1):
                # Start entry matches exactly, end entry is before the searched end
                # We use the previous entry for the end
                assert end_idx > 0, (end_idx, end, self.index[end_idx])
                assert end_idx - 1 >= start_idx, (end_idx, start_idx)
                end_entry = DateRange(*self.index[end_idx - 1])
                return slice(start_entry.offset, end_entry.offset + end_entry.length)

            case (1, 1):
                # Both entries are outside the searched range
                # We are in a gap, return an empty slice
                return slice(start_entry.offset, start_entry.offset)

            case (0, -1):
                # Start entry matches exactly, end entry is after the searched end
                # We use the current entry for the end
                assert adjust_end, "We should have adjusted the end index"
                return slice(start_entry.offset, end_entry.offset + end_entry.length)

            case (1, -1):
                # Start entry is after the searched start, end entry is after the searched end
                assert adjust_end, "We should have adjusted the end index"
                return slice(start_entry.offset, end_entry.offset + end_entry.length)

            case _:
                raise NotImplementedError(f"Case for {(diff_s, diff_e)}.")

    def boundaries(self, start: int, end: int) -> tuple[int, int]:

        start_idx = bisect.bisect_left(_Proxy(self.index), start)
        end_idx = bisect.bisect_right(_Proxy(self.index), end)
        return (self.index[start_idx], self.index[end_idx - 1])
