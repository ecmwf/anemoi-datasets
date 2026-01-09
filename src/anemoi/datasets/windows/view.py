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

from ...caching import ChunksCache
from ...date_indexing import create_date_indexing
from .annotated import AnnotatedNDArray
from .metadata import _MultipleWindowMetaData
from .metadata import _WindowMetaData
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
        store: zarr.hierarchy.Group | str,
        start_date: datetime.datetime | None = None,
        end_date: datetime.datetime | None = None,
        frequency: int | str | datetime.timedelta = 3,
        window: str | Window = "(-3,+0]",
    ) -> None:
        """Initialise a WindowView for a Zarr tabular dataset.

        Parameters
        ----------
        store : zarr.hierarchy.Group or str
            The Zarr group or path to open.
        start_date : datetime.datetime, optional
            The first date in the windowed view.
        end_date : datetime.datetime, optional
            The last date in the windowed view.
        frequency : int, str, or timedelta, default 3
            The frequency of the windowed view.
        window : str or Window, default "(-3,+0]"
            The window specification.
        """
        # Open the zarr group if a path is provided
        self.store = store if isinstance(store, zarr.hierarchy.Group) else zarr.open(store, mode="r")

        # Use provided date_indexing or create a new one for indexing
        self.date_indexing = create_date_indexing(store.attrs["date_indexing"], self.store)

        # Use a chunk cache for efficient data access
        self.data = ChunksCache(self.store["data"])

        # Determine the start and end dates for the window view
        self.start_date = to_datetime(start_date if start_date is not None else self.actual_start_end_dates[0])
        self.end_date = to_datetime(end_date if end_date is not None else self.actual_start_end_dates[1])

        if self.start_date > self.end_date:
            raise ValueError(f"WindowView: {start_date=} must be less than or equal to {end_date=}")

        # Convert frequency to timedelta and parse window if needed
        self.frequency = frequency_to_timedelta(frequency)
        self.window = window if isinstance(window, Window) else Window(window)

        assert isinstance(self.start_date, datetime.datetime)
        assert isinstance(self.end_date, datetime.datetime)
        assert isinstance(self.frequency, datetime.timedelta)

        # Compute the number of windows in the view
        last_index = (self.end_date - self.start_date) // self.frequency
        while self.start_date + last_index * self.frequency < self.end_date:
            last_index += 1
        self._len = last_index + 1

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

    def _getitem_int(self, index: any) -> np.ndarray:

        def annotate(array: np.ndarray, slice_obj: slice | None = None) -> AnnotatedNDArray:
            return AnnotatedNDArray(
                array[:, 4:],
                meta=_WindowMetaData(
                    owner=self,
                    index=index,
                    aux_array=array[:, :4],
                    slice_obj=slice_obj,
                ),
            )

        assert isinstance(index, int)
        if index < 0:
            index = self._len - index

        if not 0 <= index < self._len:
            raise IndexError(f"Index {index} out of range (len={self._len})")

        # Calculate the start and end timestamps for the window
        start = self.start_date + index * self.frequency + self.window.before
        end = self.start_date + index * self.frequency + self.window.after

        # Convert datetime to integer timestamps (seconds since epoch)
        start = round(start.timestamp())
        end = round(end.timestamp())

        # Find the boundaries in the date_indexing for the window
        first, last = self.date_indexing.boundaries(start, end)

        # assert False, (first, last, start, end)

        if first is None and last is None:
            # No data in this window, return an empty array with correct shape
            shape = (0,) + self.data.shape[1:]
            return annotate(np.zeros(shape=shape, dtype=self.data.dtype))

        first_date, (start_idx, start_cnt) = first
        last_date, (end_idx, end_cnt) = last

        print(f"WindowView: window {index}: {first_date=} {last_date=} {start=} {end=}")

        last_idx = end_idx + end_cnt

        assert first_date >= start
        assert last_date <= end

        # Exclude data at the window boundaries if specified
        if self.window.exclude_before and first_date == start:
            start_idx += start_cnt

        if self.window.exclude_after and last_date == end:
            last_idx -= end_cnt

        return annotate(self.data[start_idx:last_idx], slice(start_idx, last_idx))

    def _getitem_slice(self, index: slice) -> np.ndarray:
        start, stop, step = index.indices(self._len)
        if step != 1:
            raise ValueError("WindowView: slicing with step is not supported")

        arrays = []
        for idx in range(start, stop):
            arrays.append(self.__getitem__(idx))

        return AnnotatedNDArray(
            np.concatenate(arrays, axis=0),
            meta=_MultipleWindowMetaData(
                owner=self,
                index=index,
                children=arrays,
            ),
        )

    def _getitem_tuple(self, index: tuple) -> np.ndarray:
        result = self.__getitem__(index[0])
        return result[index[1:]]

    def __repr__(self) -> str:
        """Return a string representation of the WindowView.

        Returns
        -------
        str
            The string representation of the WindowView.
        """
        return (
            f"WindowView(start_date={self.start_date}, end_date={self.end_date}, "
            f"frequency={self.frequency}, window={self.window})"
        )

    @cached_property
    def _epochs(self) -> np.ndarray:
        """Cached property for the array of epoch timestamps corresponding to each window."""
        return np.array(
            [int((self.start_date + i * self.frequency).timestamp()) for i in range(self._len)], dtype=np.int64
        )

    @property
    def dates(self) -> np.ndarray:
        """Array of numpy.datetime64 objects for each window in the view."""
        return np.array([np.datetime64(datetime.datetime.fromtimestamp(_)) for _ in self._epochs])
