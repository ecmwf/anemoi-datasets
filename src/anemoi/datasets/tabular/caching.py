# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import zarr


class ChunksCache:
    def __init__(self, arr: zarr.Array):
        self._arr = arr
        self._chunking = arr.chunks[0]
        self._current_chunk = None
        self._cache = None
        self._offset = 0
        self._dirty = 0
        self.dtype = arr.dtype

    def __del__(self):
        self.flush()

    def __setitem__(self, key, value):

        # Same chunk
        if isinstance(key, slice):
            indices = key.indices(self._arr.shape[0])
            chunks = [_ // self._chunking for _ in indices]
            first = chunks[0]
            if all(_ == first for _ in chunks):
                self._ensure_cached(indices[0])
                key = slice(key.start - self._offset, key.stop - self._offset, key.step)
                self._cache[key] = value
                return

        if isinstance(key, slice) or (isinstance(key, tuple) and not isinstance(key[0], int)):
            # handle slices directly from the underlying array
            self.flush(True)
            self._arr[key] = value
            return

        self._ensure_cached(key)

        if isinstance(key, int):
            self._cache[key - self._offset] = value
        else:
            idx = (key[0] - self._offset,) + key[1:]
            self._cache[idx] = value

        self._dirty += 1

    def __getitem__(self, key):

        # Same chunk
        if isinstance(key, slice):
            indices = key.indices(self._arr.shape[0])
            chunks = [_ // self._chunking for _ in indices]
            first = chunks[0]
            if all(_ == first for _ in chunks):
                self._ensure_cached(indices[0])
                return self._cache[slice(key.start - self._offset, key.stop - self._offset, key.step)]

        if isinstance(key, slice) or (isinstance(key, tuple) and not isinstance(key[0], int)):
            # handle slices directly from the underlying array
            self.flush()
            # print("__getitem__ FLUSH SLICE", key)
            return self._arr[key]

        self._ensure_cached(key)

        if isinstance(key, int):
            return self._cache[key - self._offset]
        else:
            idx = (key[0] - self._offset,) + key[1:]
            return self._cache[idx]

    def flush(self, reset=False):
        """Flush cached changes to Zarr array"""
        if self._dirty > 0:
            self._arr[
                self._offset : min(
                    self._offset + self._chunking,
                    self._arr.shape[0],
                )
            ] = self._cache
            self._dirty = 0

        if reset:
            self._current_chunk = None
            self._cache = None
            self._offset = 0

    def _ensure_cached(self, key):

        if not isinstance(key, (int, tuple)):
            raise TypeError(f"Invalid key type: {type(key)}")

        if isinstance(key, tuple):
            if not isinstance(key[0], int):
                raise TypeError(f"Invalid key type: {type(key)}")
            key = key[0]

        assert key >= 0, key

        assert isinstance(key, int)
        chunk = key // self._chunking

        if self._current_chunk != chunk:
            self.flush()
            self._current_chunk = chunk
            self._cache = self._arr[
                chunk
                * self._chunking : min(
                    (chunk + 1) * self._chunking,
                    self._arr.shape[0],
                )
            ]
            self._offset = chunk * self._chunking

    @property
    def attrs(self):
        return self._arr.attrs

    @property
    def shape(self):
        return self._arr.shape

    def resize(self, *new_shape):
        self.flush(True)
        self._arr.resize(*new_shape)


class FullCache:

    def __init__(self, arr: zarr.Array):
        self.arr = arr
        self.cache = arr[:]  # Cache entire array in memory for faster access
        self.dirty = 0
        self.shape = self.cache.shape
        self.attrs = arr.attrs

    def __setitem__(self, key, value):
        self.cache[key] = value
        self.dirty += 1

    def __getitem__(self, key):
        return self.cache[key]

    def flush(self):
        """Flush cached changes to Zarr array"""
        if self.dirty > 0:
            self.arr[:] = self.cache
            self.dirty = 0

    def resize(self, new_rows, new_cols):
        self.flush()
        self.arr.resize((new_rows, new_cols))
        self.cache = np.resize(self.cache, (new_rows, new_cols))
        self.shape = self.cache.shape
        self.dirty += 1
