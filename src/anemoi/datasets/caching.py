# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import zarr
from lru import LRU

LOG = logging.getLogger(__name__)


class _Chunk:
    """Represents a chunk of a Zarr array, providing caching and synchronisation.

    Only the first dimension is considered for caching. Other chunking
    dimensions are cached together with the first dimension.

    Parameters
    ----------
    array : zarr.Array
        The Zarr array to cache chunks from.
    chunk_index : int
        The index of the chunk within the array.
    """

    def __init__(self, array: zarr.Array, chunk_index: int):
        """Initialise a chunk for the given Zarr array and chunk index.

        Parameters
        ----------
        array : zarr.Array
            The Zarr array to cache chunks from.
        chunk_index : int
            The index of the chunk within the array.
        """
        self.array = array
        self.chunk_index = chunk_index
        self.chunk_size = array.chunks[0]
        self.offset = chunk_index * self.chunk_size
        # use min to avoid indexing out of bounds of array
        self.cache = array[self.offset : min(self.offset + self.chunk_size, array.shape[0])]
        self.dirty = False
        self.lock = threading.RLock()

    def __len__(self) -> int:
        """Return the number of rows in the cached chunk.

        Returns
        -------
        int
            The number of rows in the chunk.
        """
        with self.lock:
            return len(self.cache)

    def __repr__(self) -> str:
        """Return a string representation of the chunk."""
        return f"<Chunk[offset={self.offset} shape={self.cache.shape} dirty={self.dirty}]>"

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set a value in the chunk cache and mark as dirty.

        Parameters
        ----------
        key : Any
            The key to set.
        value : Any
            The value to set.
        """
        with self.lock:
            local_index = self._local_index(key)
            self.cache[local_index] = value
            self.dirty = True

    def __getitem__(self, key: Any) -> Any:
        """Get a value from the chunk cache.

        Parameters
        ----------
        key : Any
            The key to access.

        Returns
        -------
        Any
            The value at the given key.
        """
        with self.lock:
            local_index = self._local_index(key)
            return self.cache[local_index]

    def _local_index(self, key: Any) -> Any:
        """Checks and converts the global key to a local index within the chunk.

        Parameters
        ----------
        key : Any
            The global key to convert.

        Returns
        -------
        Any
            The local index within the chunk.

        Raises
        ------
        IndexError
            If the key is out of chunk bounds.
        TypeError
            If the key type is unsupported.
        """
        match key:
            case int():
                local_index = key - self.offset
                if not (0 <= local_index < len(self.cache)):
                    raise IndexError("Index out of chunk bounds")
                return local_index

            case slice():
                # Assumes that the slice is normalised
                local_index = slice(key.start - self.offset, key.stop - self.offset, key.step)
                if not (0 <= local_index.start < len(self.cache)) or not (0 <= local_index.stop <= len(self.cache)):
                    raise IndexError("Slice out of chunk bounds")
                return local_index

            case tuple():
                return (self._local_index(key[0]),) + key[1:]

            case _:
                raise TypeError(f"Unsupported key type: {type(key)}")

    def flush(self) -> None:
        """Flush cached changes to Zarr array."""
        with self.lock:
            if self.dirty:
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.debug(f"Flushing chunk {self.chunk_index} {self}")
                self.array[self.offset : self.offset + len(self.cache)] = self.cache
                self.dirty = False

    def _remove(self, lru: LRU, why: str) -> None:
        """Remove this chunk from the LRU cache, flushing if necessary.

        Parameters
        ----------
        lru : LRU
            The LRU cache.
        why : str
            Reason for removal.
        """
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Removing chunk {self.chunk_index} {self} because {why}")
        self.flush()
        del lru[self.chunk_index]

    def resize(self, lru: LRU, new_array_shape: tuple[int, ...]) -> None:
        """Handles resizing of the underlying array.

        Parameters
        ----------
        lru : LRU
            The LRU cache.
        new_array_shape : tuple of int
            The new shape of the array.
        """
        with self.lock:
            if len(self.cache.shape) != len(new_array_shape):
                return self._remove(
                    lru,
                    "Size mismatch resizing shape from ",
                    self.cache.shape,
                    " to ",
                    new_array_shape,
                )

            if len(new_array_shape) > 1 and self.cache.shape[2:] != new_array_shape[2:]:
                return self._remove(
                    lru,
                    f"Shape mismatch resizing shape from {self.cache.shape} to {new_array_shape}",
                )

            last_idx = self.offset + self.cache.shape[0]

            if not (last_idx <= new_array_shape[0]):
                return self._remove(lru, f"Chunk exceeds new array size {new_array_shape}")

            if last_idx != min(self.offset + self.chunk_size, new_array_shape[0]):
                return self._remove(lru, f"Chunk size mismatch after resizing array to {new_array_shape}")


class ChunksCache:
    """Caches chunks of a Zarr array for efficient access and modification."""

    def __init__(self, array: zarr.Array, chunk_caching: int = 512 * 1024 * 1024, read_ahead: bool = False):
        """Initialise the chunk cache for a Zarr array.

        Parameters
        ----------
        array : zarr.Array
            The Zarr array to cache.
        chunk_caching : int, optional
            The cache size in bytes (default is 512 * 1024 * 1024).
        read_ahead : bool, optional
            Whether to enable read-ahead (default is False).
        """
        self._arr = array
        self._nrows_in_chunks = array.chunks[0]

        size_per_row = np.dtype(array.dtype).itemsize * array[0].size
        chunk_size = self._nrows_in_chunks * size_per_row
        chunk_caching = max(chunk_caching, chunk_size)
        chunks_in_cache = chunk_caching // chunk_size

        LOG.info(
            f"Initializing ChunksCache with chunk shape {array.chunks}, "
            f"caching {chunks_in_cache} chunks ({chunks_in_cache * chunk_size / 1024 / 1024:.2f} MB)",
        )

        self._lru_chunks_cache = LRU(chunks_in_cache, callback=self._evict_chunk)
        self._lock = threading.RLock()

        self._last_read_ahead_chunk = 0

        if read_ahead:
            self._read_ahead = ThreadPoolExecutor(max_workers=1)
            self._last_read_ahead_chunk = 0
            self._read_ahead.submit(self._read_ahead_worker, 0)
        else:
            self._read_ahead = None

    def _read_ahead_worker(self, chunk_number: int) -> None:
        """Worker function for read-ahead, loads the next chunk in the background.

        Parameters
        ----------
        chunk_number : int
            The chunk index to read ahead.
        """
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Read-ahead loading chunk {chunk_number}")
            if chunk_number > self._max_chunk_index:
                return
        self._ensure_chunk_in_cache(chunk_number, from_read_ahead=True)

    @staticmethod
    def _evict_chunk(key: int, chunk: _Chunk) -> None:
        """Callback called when a chunk is evicted from the cache.

        Parameters
        ----------
        key : int
            The chunk index.
        chunk : _Chunk
            The chunk to evict.
        """
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Evicting chunk {key} {chunk}")

        chunk.flush()

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set a value in the cached array.

        Parameters
        ----------
        key : Any
            The key to set.
        value : Any
            The value to set.
        """
        with self._lock:
            chunk, key = self._get_key_chunk(key)
            chunk[key] = value

    def __getitem__(self, key: Any) -> Any:
        """Get a value from the cached array.

        Parameters
        ----------
        key : Any
            The key to access.

        Returns
        -------
        Any
            The value at the given key.
        """

        with self._lock:
            chunk, key = self._get_key_chunk(key)
            return chunk[key]

    def flush(self) -> None:
        """Flush all cached chunks to the underlying Zarr array."""
        with self._lock:
            for chunk in self._lru_chunks_cache.values():
                chunk.flush()

    @property
    def _max_chunk_index(self) -> int:
        """The maximum chunk index based on the array size."""
        return (self._arr.shape[0] - 1) // self._nrows_in_chunks

    def _ensure_chunk_in_cache(self, chunk_index: int, from_read_ahead: bool = False) -> _Chunk:
        """Ensure a chunk is present in the cache.

        Parameters
        ----------
        chunk_index : int
            The index of the chunk.

        from_read_ahead : bool, optional
            Whether this call is from the read-ahead worker (default is False).

        Returns
        -------
        _Chunk
            The cached chunk.
        """
        with self._lock:
            if chunk_index in self._lru_chunks_cache:
                return self._lru_chunks_cache[chunk_index]

        # Create the chunk outside the lock so we don't block other operations while loading from disk
        # In the unlikely event of multiple threads loading the same chunk, the LRU cache will handle duplicates
        # and the garbage collector will clean up unreferenced chunks
        chunk = _Chunk(self._arr, chunk_index)

        with self._lock:
            self._lru_chunks_cache[chunk_index] = chunk

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(
                    f"Loaded chunk {chunk_index} {self._lru_chunks_cache[chunk_index]}. Cached: {sorted(self._lru_chunks_cache.keys())},"
                    f" {sum(chunk.cache.nbytes for chunk in self._lru_chunks_cache.values()) / 1024 / 1024:.2f} MB"
                    f" dirty={sum(chunk.dirty for chunk in self._lru_chunks_cache.values())}",
                )

            if self._read_ahead is not None and not from_read_ahead:
                with self._lock:
                    if chunk_index + 1 > self._last_read_ahead_chunk:
                        self._last_read_ahead_chunk = chunk_index + 1
                        self._read_ahead.submit(self._read_ahead_worker, chunk_index + 1)

            return self._lru_chunks_cache[chunk_index]

    def _normalise_slice(self, key: slice) -> slice:
        """Convert a slice to absolute indices (no Nones, no negatives).

        Parameters
        ----------
        key : slice
            The slice to normalise.

        Returns
        -------
        slice
            The normalised slice.
        """
        result = slice(*key.indices(self._arr.shape[0]))
        assert result.step >= 1, "Negative or zero step slices are not supported"
        assert result.start <= result.stop, "Slice start must be less than or equal to stop"
        assert 0 <= result.start, "Slice start negative after normalisation"
        return result

    def _get_key_chunk(self, key: Any) -> tuple[Any, Any]:
        """Retrieves the appropriate chunk and a normalised key.

        Parameters
        ----------
        key : Any
            The key to access.

        Returns
        -------
        tuple
            The chunk and the normalised key.

        Raises
        ------
        TypeError
            If the key type is unsupported.
        """
        match key:

            case int():
                if key < 0:
                    key += self._arr.shape[0]
                chunk_index = key // self._nrows_in_chunks
                return self._ensure_chunk_in_cache(chunk_index), key

            case slice():
                return self._get_key_chunk((key, slice(None, None, None)))

            case tuple():
                if isinstance(key[0], int):
                    chunk_index = key[0] // self._nrows_in_chunks
                    return self._ensure_chunk_in_cache(chunk_index), key

                if isinstance(key[0], slice):
                    slice_0 = self._normalise_slice(key[0])
                    key = (slice_0,) + key[1:]

                    start, stop, step = slice_0.start, slice_0.stop, slice_0.step

                    if start == stop:
                        # Handle empty slice
                        in_lru = list(self._lru_chunks_cache.keys())
                        chunk_index = in_lru[0] if in_lru else 0
                        return self._ensure_chunk_in_cache(chunk_index), key

                    last = start + step * ((stop - 1 - start) // step)
                    start_chunk = start // self._nrows_in_chunks
                    last_chunk = last // self._nrows_in_chunks

                    if start_chunk == last_chunk:
                        # The index is within a single chunk
                        return self._ensure_chunk_in_cache(start_chunk), key

                    # The index spans multiple chunks, use multi-chunk handler
                    return _MultiChunkSlice(self, key), key

                raise TypeError(f"Unsupported key type in tuple: {type(key[0])} ({key[0]})")

            case _:
                raise TypeError(f"Unsupported key type: {type(key)} ({key})")

    @property
    def attrs(self):
        """The attributes of the underlying Zarr array."""
        return self._arr.attrs

    @property
    def shape(self):
        """The shape of the underlying Zarr array."""
        return self._arr.shape

    @property
    def dtype(self):
        """The dtype of the underlying Zarr array."""
        return self._arr.dtype

    @property
    def chunks(self):
        """The chunk shape of the underlying Zarr array."""
        return self._arr.chunks

    def __len__(self) -> int:
        """Return the number of elements in the underlying array."""
        return len(self._arr)

    def resize(self, *new_shape: int) -> None:
        """Resize the underlying Zarr array and adjust cached chunks.

        Parameters
        ----------
        *new_shape : int
            The new shape of the array.
        """
        with self._lock:
            for chunk in self._lru_chunks_cache.values():
                chunk.resize(self._lru_chunks_cache, new_shape)
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(f"Resized underlying array from {self._arr.shape} to {new_shape}")
            self._arr.resize(*new_shape)

    def __enter__(self) -> "ChunksCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()


class _MultiChunkSlice:
    """Handles multi-chunk slicing for ChunksCache."""

    def __init__(self, cache: ChunksCache, key: tuple[Any, ...]):
        """Initialise a multi-chunk slice handler.

        Parameters
        ----------
        cache : ChunksCache
            The chunk cache.
        key : tuple
            The key representing the slice.
        """
        self._cache = cache
        self._key = key
        assert isinstance(key, tuple) and isinstance(key[0], slice)

    def __setitem__(self, key: tuple[Any, ...], value: Any) -> None:
        """Set values across multiple chunks.

        Parameters
        ----------
        key : tuple
            The key representing the slice.
        value : Any
            The value to set.
        """
        for chunk, idx, shape in self._split_chunks(key):
            chunk[idx] = value[shape]

    def __getitem__(self, key: tuple[Any, ...]) -> Any:
        """Get values across multiple chunks.

        Parameters
        ----------
        key : tuple
            The key representing the slice.

        Returns
        -------
        Any
            The concatenated values from all relevant chunks.
        """
        values = []
        for chunk, idx, shape in self._split_chunks(key):
            values.append(chunk[idx][shape])
        return np.concatenate(values, axis=0)

    def _split_chunks(self, key: tuple[Any, ...]) -> Iterator[tuple[_Chunk, tuple[Any, ...], slice]]:
        """Splits the key into per-chunk slices.

        Parameters
        ----------
        key : tuple
            The key representing the slice.

        Yields
        ------
        tuple
            The chunk, the index for that chunk, and the slice for assembling the result.
        """
        assert self._key == key
        slice0 = key[0]

        # Assume slice is normalised
        start, stop, step = slice0.start, slice0.stop, slice0.step

        # First and last indices selected by the slice
        first_idx = start
        last_idx = start + ((stop - start - 1) // step) * step

        chunk_size = self._cache._nrows_in_chunks

        # First and last chunks touched
        first_chunk = first_idx // chunk_size
        last_chunk = last_idx // chunk_size

        chunks = {}
        for chunk_index in range(first_chunk, last_chunk + 1):
            chunk_start = chunk_index * chunk_size
            chunk_end = chunk_start + chunk_size

            # Find first index in this chunk that the slice selects
            if chunk_start <= start:
                global_first = start
            else:
                # Need start + k*step >= chunk_start
                k = (chunk_start - start + step - 1) // step  # Ceiling division
                global_first = start + k * step

            # Check if this index is still in range
            if global_first >= stop or global_first >= chunk_end:
                continue

            # Find last index in this chunk that the slice selects
            effective_stop = min(stop, chunk_end)
            global_last = start + ((effective_stop - start - 1) // step) * step

            chunks[chunk_index] = slice(global_first, global_last + 1, step)

        shape = 0
        for chunk_index, s in sorted(chunks.items()):
            chunk = self._cache._ensure_chunk_in_cache(chunk_index)
            end = min(len(chunk), self._cache._nrows_in_chunks, (s.stop - s.start) // s.step)
            yield chunk, (s,) + key[1:], slice(shape, shape + end)
            shape += end
