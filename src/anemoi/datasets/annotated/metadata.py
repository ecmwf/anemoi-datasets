# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np

from anemoi.datasets.epochs import array_to_epoch
from anemoi.datasets.epochs import epoch_to_date
from anemoi.datasets.epochs import epochs_to_datetime64

from .base import WindowMetaDataBase

LOG = logging.getLogger(__name__)


class WindowMetaData(WindowMetaDataBase):
    """Holds metadata for a windowed data array, including reference to the owner,
    index, and auxiliary array for date and location information.
    """

    def __init__(self, owner, index, aux_array, slice_obj) -> None:
        """Initialise the _WindowMetaData object.

        Parameters
        ----------
        owner : WindowView
            The WindowView instance that owns this metadata.
        index : int
            The window index.
        aux_array : np.ndarray
            Auxiliary array containing date and location information.
        """
        super().__init__(owner, index)

        self._aux_array = aux_array
        self._slice_obj = slice_obj  # for debugging purposes

    @property
    def latitudes(self) -> np.ndarray:
        """Array of latitudes for the window."""
        return self._aux_array[:, 2]

    @property
    def longitudes(self) -> np.ndarray:
        """Array of longitudes for the window."""
        return self._aux_array[:, 3]

    @property
    def dates(self) -> np.ndarray:
        """Array of dates for the window."""
        return epochs_to_datetime64(array_to_epoch(self._aux_array))

    @property
    def timedeltas(self) -> np.ndarray:
        """Array of time deltas for the window."""
        return (array_to_epoch(self._aux_array) - self.owner._epochs[self.index]).astype("float32")

    @property
    def reference_date(self) -> np.datetime64:
        """The reference date for the window."""
        return np.datetime64(epoch_to_date(self.owner._epochs[self.index]))

    @property
    def reference_dates(self) -> np.ndarray:
        """The reference date for the window."""
        return np.array([self.reference_date])

    @property
    def boundaries(self) -> list[slice]:
        return [slice(0, len(self._aux_array))]

    @property
    def _unsharded_window_size(self) -> int:
        """Row count of this window before sharding."""
        range_slice = self.owner._slice(self.index)
        return range_slice.stop - range_slice.start

    @property
    def unsharded_boundaries(self) -> list[slice]:
        """Where this shard's rows sit within the unsharded window.

        Returns a single slice ``[lo, hi)`` giving the position this shard's
        rows occupy in the full (unsharded) window ``index``. For an unsharded
        view this equals :attr:`boundaries`. Use it to scatter a shard's rows
        into an array sized for the whole window.
        """
        n = self.owner.num_shards
        i = self.owner.shard_index or 0
        total = self._unsharded_window_size
        lo = (total * i) // n
        hi = (total * (i + 1)) // n
        return [slice(lo, hi)]


class MultipleWindowMetaData(WindowMetaDataBase):
    """Holds metadata for multiple windowed data arrays, aggregating metadata from child arrays."""

    def __init__(self, owner, index, children) -> None:
        """Initialise the _MultipleWindowMetaData object.

        Parameters
        ----------
        owner : WindowView
            The WindowView instance that owns this metadata.
        index : slice
            The slice index.
        children : list[_WindowMetaData]
            List of child metadata objects.
        """
        super().__init__(owner, index)
        self.children = children

    @property
    def latitudes(self) -> np.ndarray:
        """Array of latitudes for the multiple windows."""
        return np.concatenate([child.latitudes for child in self.children], axis=0)

    @property
    def longitudes(self) -> np.ndarray:
        """Array of longitudes for the multiple windows."""
        return np.concatenate([child.longitudes for child in self.children], axis=0)

    @property
    def dates(self) -> np.ndarray:
        """Array of dates for the multiple windows."""
        return np.concatenate([child.dates for child in self.children], axis=0)

    @property
    def timedeltas(self) -> np.ndarray:
        """Array of time deltas for the multiple windows."""
        return np.concatenate([child.timedeltas for child in self.children], axis=0)

    @property
    def reference_date(self) -> np.datetime64:
        """The reference date for the first window."""
        refs = self.reference_dates
        if len(refs) == 1:
            return refs[0]
        raise ValueError(f"MultipleWindowMetaData: reference_date is ambiguous for multiple windows {refs}")

    @property
    def reference_dates(self) -> np.ndarray:
        """The reference date for the first window."""
        return np.concatenate([child.reference_dates for child in self.children], axis=0)

    @property
    def boundaries(self) -> list[slice]:
        result = []
        offset = 0
        for child in self.children:
            length = len(child)
            result.append(slice(offset, offset + length))
            offset += length
        return result

    @property
    def unsharded_boundaries(self) -> list[slice]:
        """Per-window placement of this shard's rows in the unsharded result.

        One slice per window. The offsets accumulate the *unsharded* window
        sizes, so the slices index into an array sized for the whole
        (unsharded) concatenation of these windows. Pair with
        :attr:`boundaries` (which indexes this shard's own array) to scatter a
        shard's rows into the full array.
        """
        result = []
        offset = 0
        for child in self.children:
            meta = child._anemoi_annotation
            (within_window,) = meta.unsharded_boundaries
            result.append(slice(offset + within_window.start, offset + within_window.stop))
            offset += meta._unsharded_window_size
        return result
