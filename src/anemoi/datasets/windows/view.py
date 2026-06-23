# (C) Copyright 2025 Anemoi contributors.
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
import zarr
from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.utils.dates import to_datetime

from anemoi.datasets.usage.misc import as_first_date
from anemoi.datasets.usage.misc import as_last_date

from ..annotated._numpy import AnnotatedNDArray
from ..annotated.metadata import MultipleWindowMetaData
from ..annotated.metadata import WindowMetaData
from ..buffering import RandomReadBuffer
from ..date_indexing import create_date_indexing
from ..epochs import date_to_epoch
from ..epochs import epoch_to_date
from .window import Window

LOG = logging.getLogger(__name__)


class WindowView:
    """Provides a view into a Zarr tabular dataset with windowed, frequency-based access.

    Allows subsetting of a dataset by time window, frequency, and start/end dates. Handles conversion
    between date indices and actual data slices. Provides methods to adjust the window, frequency, and
    date range, returning new WindowView instances. Implements __getitem__ for windowed data access,
    including filtering and time delta calculation. Caches epochs and provides date arrays for the windowed view.
    """

    def __init__(
        self,
        store: zarr.Group | str,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        frequency: int | str | datetime.timedelta = 3,
        window: str | Window = "(-3,+0]",
        shard: tuple[int, int] | None = None,
    ) -> None:
        """Initialise a WindowView for a Zarr tabular dataset.

        Parameters
        ----------
        store : zarr.Group or str
            The Zarr group or path to open.
        start_date : datetime.datetime, optional
            The first date in the windowed view.
        end_date : datetime.datetime, optional
            The last date in the windowed view.
        frequency : int, str, or timedelta, default 3
            The frequency of the windowed view.
        window : str or Window, default "(-3,+0]"
            The window specification.
        shard : tuple of (int, int), optional
            A ``(index, count)`` pair selecting one of ``count`` shards. Each
            window's rows are partitioned into ``count`` contiguous,
            non-overlapping pieces and only piece ``index`` is returned. The
            window count, dates, and frequency are unchanged. ``None`` (default)
            returns the whole window.
        """
        # Open the zarr group if a path is provided
        self.store = store if isinstance(store, zarr.Group) else zarr.open(store, mode="r")

        # Use provided date_indexing or create a new one for indexing
        self.date_indexing = create_date_indexing(store.attrs["date_indexing"], self.store)

        # Use a chunk cache for efficient data access
        self.data = RandomReadBuffer(self.store["data"])

        # Determine the start and end dates for the window view
        self.start_date = to_datetime(start_date if start_date is not None else self.actual_start_end_dates[0])
        self.end_date = to_datetime(end_date if end_date is not None else self.actual_start_end_dates[1])

        # if self.start_date > self.end_date:
        #     raise ValueError(f"WindowView: {start_date=} must be less than or equal to {end_date=}")

        # Convert frequency to timedelta and parse window if needed
        self.frequency = frequency_to_timedelta(frequency)
        self.window = window if isinstance(window, Window) else Window(window)

        # Sharding: (index, count) or None for the whole window
        if shard is not None:
            index, count = shard
            if not (isinstance(count, int) and count >= 1):
                raise ValueError(f"WindowView: shard count must be a positive integer, got {count!r}")
            if not (isinstance(index, int) and 0 <= index < count):
                raise ValueError(f"WindowView: shard index must be in [0, {count}), got {index!r}")
        self.shard = shard

        assert isinstance(self.start_date, datetime.datetime)
        assert isinstance(self.end_date, datetime.datetime)
        assert isinstance(self.frequency, datetime.timedelta)

    def set_start(self, start: datetime.datetime) -> "WindowView":
        """Return a new WindowView with the start date aligned to the frequency using as_first_date.

        Parameters
        ----------
        start : datetime.datetime
            The new start date to align.

        Returns
        -------
        WindowView
            A new WindowView instance with the updated start date.
        """
        # TODO: check if in-line with the way we implemented that logic for fields
        return WindowView(
            store=self.store,
            start_date=as_first_date(start, None, frequency=self.frequency),
            end_date=self.end_date,
            frequency=self.frequency,
            window=self.window,
            shard=self.shard,
        )

    def set_end(self, end: datetime.datetime) -> "WindowView":
        """Return a new WindowView with the end date aligned to the frequency using as_last_date.

        Parameters
        ----------
        end : datetime.datetime
            The new end date to align.

        Returns
        -------
        WindowView
            A new WindowView instance with the updated end date.
        """
        # TODO: check if in-line with the way we implemented that logic for fields
        return WindowView(
            store=self.store,
            start_date=self.start_date,
            end_date=as_last_date(end, None, frequency=self.frequency),
            frequency=self.frequency,
            window=self.window,
            shard=self.shard,
        )

    def set_frequency(self, frequency: str | int | datetime.timedelta) -> "WindowView":
        """Return a new WindowView with the updated frequency.

        Parameters
        ----------
        frequency : str, int, or datetime.timedelta
            The new frequency for the windowed view.

        Returns
        -------
        WindowView
            A new WindowView instance with the updated frequency.
        """
        return WindowView(
            store=self.store,
            start_date=self.start_date,
            end_date=self.end_date,
            frequency=frequency,
            window=self.window,
            shard=self.shard,
        )

    def set_window(self, window: str | Window) -> "WindowView":
        """Return a new WindowView with the updated window specification.

        Parameters
        ----------
        window : str or Window
            The new window specification.

        Returns
        -------
        WindowView
            A new WindowView instance with the updated window.
        """
        return WindowView(
            store=self.store,
            start_date=self.start_date,
            end_date=self.end_date,
            frequency=self.frequency,
            window=window,
            shard=self.shard,
        )

    def _internal_set_shard(self, index: int, count: int) -> "WindowView":
        """Return a new WindowView that exposes only one shard of each window.

        Each window's rows are partitioned into ``count`` contiguous,
        non-overlapping pieces; the returned view yields only piece ``index``.

        Parameters
        ----------
        index : int
            The shard index, in ``[0, count)``.
        count : int
            The total number of shards.

        Returns
        -------
        WindowView
            A new WindowView restricted to the requested shard.

        Raises
        ------
        ValueError
            If this view is already sharded (sharding cannot be chained).
        """
        if self.shard is not None:
            raise ValueError(f"WindowView: sharding cannot be chained (already sharded with {self.shard})")
        return WindowView(
            store=self.store,
            start_date=self.start_date,
            end_date=self.end_date,
            frequency=self.frequency,
            window=self.window,
            shard=(index, count),
        )

    @cached_property
    def actual_start_end_dates(self) -> tuple[datetime.datetime, datetime.datetime]:
        """The actual start and end dates available in the underlying dataset."""
        return self.date_indexing.start_end_dates()

    def __len__(self) -> int:
        """Return the number of windows in the view.

        Returns
        -------
        int
            The number of windows in the view.
        """

        return self._len

    @property
    def _len(self) -> int:

        # Compute the number of windows in the view
        # TODO: review that logic

        last_index = (self.end_date - self.start_date) // self.frequency
        while self.start_date + last_index * self.frequency <= self.end_date:
            last_index += 1

        return last_index + 1

    # ------------------------------------------------------------------
    # Sharding introspection
    #
    # These describe how each window's rows are (or would be) partitioned
    # across shards. They depend only on the *unsharded* window sizes and the
    # number of shards, so they are identical on every shard of the same view
    # and never read the data itself (only the date index).
    # ------------------------------------------------------------------

    @property
    def shard_index(self) -> int | None:
        """The index of this shard, or ``None`` if the view is not sharded."""
        return self.shard[0] if self.shard is not None else None

    @property
    def num_shards(self) -> int:
        """The number of shards (``1`` if the view is not sharded)."""
        return self.shard[1] if self.shard is not None else 1

    @cached_property
    def unsharded_sizes(self) -> np.ndarray:
        """Row count of every window in the *unsharded* dataset.

        One entry per window; ``unsharded_sizes[n]`` is the number of rows
        window ``n`` holds before sharding. Computed from the date index only.
        """
        sizes = np.empty(self._len, dtype=np.int64)
        for i in range(self._len):
            range_slice = self._slice(i)
            sizes[i] = range_slice.stop - range_slice.start
        return sizes

    @cached_property
    def shard_sizes(self) -> np.ndarray:
        """Row counts per window per shard, shaped ``(num_windows, num_shards)``.

        ``shard_sizes[n, i]`` is the number of rows shard ``i`` holds for
        window ``n``. Rows sum to :attr:`unsharded_sizes` (``sum(axis=1)``).
        """
        n = self.num_shards
        sizes = self.unsharded_sizes
        # bounds[n, i] = sizes[n] * i // num_shards, for i in 0..num_shards
        bounds = np.stack([(sizes * i) // n for i in range(n + 1)], axis=1)
        return bounds[:, 1:] - bounds[:, :-1]

    @property
    def total_size(self) -> int:
        """Total number of rows in the *unsharded* dataset across all windows."""
        return int(self.unsharded_sizes.sum())

    def __getitem__(self, index: Any) -> np.ndarray:
        """Retrieve the data for the specified window index, applying window boundaries and filtering.

        Parameters
        ----------
        index : Any
            The window index to retrieve.

        Returns
        -------
        np.ndarray
            The filtered data array for the specified window.
        """

        match index:
            case int():
                return self._getitem_int(index)

            case slice():
                return self._getitem_slice(index)

            case tuple():
                return self._getitem_tuple(index)

            case _:
                raise TypeError(f"WindowView: invalid index type: {type(index)}")

    def _slice(self, index: int) -> np.ndarray:
        assert isinstance(index, int)
        if index < 0:
            index = self._len - index

        if not 0 <= index < self._len:
            raise IndexError(f"Index {index} out of range (len={self._len})")

        # Calculate the start and end timestamps for the window
        query_start = self.start_date + index * self.frequency + self.window.before
        query_end = self.start_date + index * self.frequency + self.window.after

        query_start_save = query_start
        query_end_save = query_end

        # Convert datetime to integer timestamps (seconds since epoch)
        query_start = round(date_to_epoch(query_start))
        query_end = round(date_to_epoch(query_end))

        assert abs(epoch_to_date(query_start) - query_start_save) < datetime.timedelta(seconds=1), (
            query_start,
            query_start_save,
            epoch_to_date(query_start),
        )
        assert abs(epoch_to_date(query_end) - query_end_save) < datetime.timedelta(seconds=1), (
            query_end,
            query_end_save,
            epoch_to_date(query_end),
        )

        # Because the accuracy is the second, we can adjust the query
        # to exclude the boundaries if needed. THe index must return the range of
        # data including the boundaries, so we adjust the query accordingly.

        if self.window.exclude_before:
            query_start += 1

        if self.window.exclude_after:
            query_end -= 1

        range_slice = "(not set)"

        # Find the boundaries in the date_indexing for the window
        try:
            range_slice = self.date_indexing.range_search(query_start, query_end, len(self.data))

            assert range_slice.step in (None, 1), range_slice.step
            assert range_slice.start >= 0, range_slice.start
            assert range_slice.stop >= range_slice.start, range_slice
            assert range_slice.stop <= len(self.data), (range_slice, len(self.data))

            return range_slice

        except (IndexError, StopIteration) as e:
            # We don't have that error to stop iterations in the caller
            # We should be in control here
            raise ValueError(f"Error retrieving data for window index {index}, slice={range_slice}: {e}") from e

    def _apply_shard(self, range_slice: slice) -> slice:
        """Narrow a window's row range to the configured shard.

        The ``[start, stop)`` range is partitioned into ``count`` contiguous,
        non-overlapping pieces that exactly cover the original range, and the
        slice for piece ``index`` is returned. Returns ``range_slice`` unchanged
        when no shard is configured.
        """
        if self.shard is None:
            return range_slice

        index, count = self.shard
        start, stop = range_slice.start, range_slice.stop
        total = stop - start
        lo = start + (total * index) // count
        hi = start + (total * (index + 1)) // count
        return slice(lo, hi)

    def _getitem_int(self, index: any) -> np.ndarray:

        def annotate(array: np.ndarray, slice_obj: slice | None = None) -> AnnotatedNDArray:
            return AnnotatedNDArray(
                array[:, 4:],
                anemoi_annotation=WindowMetaData(
                    owner=self,
                    index=index,
                    aux_array=array[:, :4],
                    slice_obj=slice_obj,
                ),
            )

        range_slice = self._apply_shard(self._slice(index))

        try:
            # Find the boundaries in the date_indexing for the window
            values = self.data[range_slice]
        except (IndexError, StopIteration) as e:
            # We don't have that error to stop iterations in the caller
            # We should be in control here
            raise ValueError(f"Error retrieving data for window index {index}, slice={range_slice}: {e}") from e

        assert values.shape[0] == range_slice.stop - range_slice.start, (
            values.shape,
            range_slice,
            range_slice.stop - range_slice.start,
        )

        return annotate(values, range_slice)

    def _getitem_slice(self, index: slice) -> np.ndarray:
        start, stop, step = index.indices(self._len)
        if step != 1:
            raise ValueError("WindowView: slicing with step is not supported")

        arrays = []
        for idx in range(start, stop):
            arrays.append(self.__getitem__(idx))

        return AnnotatedNDArray(
            np.concatenate(arrays, axis=0),
            anemoi_annotation=MultipleWindowMetaData(
                owner=self,
                index=index,
                children=arrays,
            ),
        )

    def _getitem_tuple(self, index: tuple) -> np.ndarray:
        result = self.__getitem__(index[0])
        return result[:, *index[1:]]

    def __repr__(self) -> str:
        """Return a string representation of the WindowView.

        Returns
        -------
        str
            The string representation of the WindowView.
        """
        return (
            f"WindowView(start_date={self.start_date}, end_date={self.end_date}, "
            f"frequency={self.frequency}, window={self.window}, shard={self.shard})"
        )

    @cached_property
    def _epochs(self) -> np.ndarray:
        """Cached property for the array of epoch timestamps corresponding to each window."""
        return np.array(
            [int(date_to_epoch(self.start_date + i * self.frequency)) for i in range(self._len)],
            dtype=np.int64,
        )

    @property
    def dates(self) -> np.ndarray:
        """Array of numpy.datetime64 objects for each window in the view."""
        return np.array([np.datetime64(epoch_to_date(_)) for _ in self._epochs])

    @property
    def data_length(self) -> int:
        """Return the total length of the underlying data.
        For debugging purposes.
        """
        return len(self.data)

    @property
    def whole_range(self) -> slice:
        """Return slice of the whole range in the underlying data.
        For debugging purposes.
        """

        query_start = self._epochs[0] + int(self.window.before.total_seconds())
        if self.window.exclude_before:
            query_start += 1

        query_end = self._epochs[-1] + int(self.window.after.total_seconds())
        if self.window.exclude_after:
            query_end -= 1

        result = self.date_indexing.range_search(query_start, query_end, len(self.data))

        actual_start, actual_end = self.actual_start_end_dates

        if self.start_date <= actual_start:
            assert result.start == 0, (self.start_date, actual_start, result)

        if self.end_date >= actual_end:
            assert result.stop == len(self.data), (self.end_date, actual_end, result, len(self.data))

        return result

    def plot_dates(self):
        dates = []
        counts = []
        for i in range(len(self)):
            range_slice = self._slice(i)
            dates.append(self.start_date + i * self.frequency)
            counts.append(range_slice.stop - range_slice.start)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.bar(dates, counts)
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.title("Counts per Date Bucket")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
