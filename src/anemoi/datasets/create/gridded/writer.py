# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


class ViewCacheArray:
    """A class that provides a caching mechanism for writing to a NumPy-like array.

    The is initialised with a NumPy-like array, a shape and a list to reindex the first
    dimension. The array is used to store the final data, while the cache is used to
    temporarily store the data before flushing it to the array.

    The `flush` method copies the contents of the cache to the final array.
    """

    def __init__(self, array: NDArray[Any], *, shape: tuple[int, ...], indexes: list[int]):
        """Initialize the ViewCacheArray.

        Parameters
        ----------
        array : NDArray[Any]
            The NumPy-like array to store the final data.
        shape : tuple[int, ...]
            The shape of the cache array.
        indexes : list[int]
            List to reindex the first dimension.
        """
        assert len(indexes) == shape[0], (len(indexes), shape[0])
        self.array = array
        self.dtype = array.dtype
        self.cache = np.full(shape, np.nan, dtype=self.dtype)
        self.indexes = indexes

    def __setitem__(self, key: tuple[int, ...], value: NDArray[Any]) -> None:
        """Set the value in the cache array at the specified key.

        Parameters
        ----------
        key : tuple[int, ...]
            The index key to set the value.
        value : NDArray[Any]
            The value to set in the cache array.
        """
        self.cache[key] = value

    def flush(self) -> None:
        """Copy the contents of the cache to the final array."""
        for i in range(self.cache.shape[0]):
            global_i = self.indexes[i]
            self.array[global_i] = self.cache[i]
