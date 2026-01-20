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


class _Buffer:
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
                assert key >= 0, f"Only positive indices are supported {key}"
                local_index = key - self.offset
                if not (0 <= local_index < len(self.cache)):
                    raise IndexError("Index out of chunk bounds")
                return local_index

            case slice():
                assert key.step >= 1, "Negative or zero step slices are not supported"
                assert key.start is not None and key.stop is not None, "Slice start and stop cannot be None"

                # Assumes that the slice is normalised
                local_index = slice(key.start - self.offset, key.stop - self.offset, key.step)
                if not (0 <= local_index.start < len(self.cache)) or not (0 <= local_index.stop <= len(self.cache)):
                    raise IndexError("Slice out of chunk bounds")
                return local_index

            case tuple():
                return (self._local_index(key[0]),) + key[1:]

            case np.ndarray():
                local_index = key - self.offset
                if any(not (0 <= li < len(self.cache)) for li in local_index):
                    raise IndexError(f"Array indices out of chunk bounds {key} => {local_index} ({self})")
                return local_index

            case list():
                return self._local_index(np.array(key))

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

            if len(new_array_shape) > 1 and self.cache.shape[1:] != new_array_shape[1:]:
                return self._remove(
                    lru,
                    f"Shape mismatch resizing shape from {self.cache.shape} to {new_array_shape}",
                )

            last_idx = self.offset + self.cache.shape[0]

            if not (last_idx <= new_array_shape[0]):
                return self._remove(lru, f"Chunk exceeds new array size {new_array_shape}")

            if last_idx != min(self.offset + self.chunk_size, new_array_shape[0]):
                return self._remove(lru, f"Chunk size mismatch after resizing array to {new_array_shape}")


class ReadAheadWriteBehindBuffer:
    """Caches chunks of a Zarr array for efficient access and modification."""

    def __init__(
        self,
        array: zarr.Array,
        buffer_size: int = 512 * 1024 * 1024,
        max_cached_chunks: int = None,
        no_reload: bool = False,
    ):
        """Initialise the chunk cache for a Zarr array.

        Parameters
        ----------
        array : zarr.Array
            The Zarr array to cache.
        buffer_size : int, optional
            The cache size in bytes (default is 512 * 1024 * 1024).
        read_ahead : bool, optional
            Whether to enable read-ahead (default is False).
        """
        self._arr = array
        self._nrows_in_chunks = array.chunks[0]
        self._no_reload = no_reload
        self._was_loaded = set()

        size_per_row = np.dtype(array.dtype).itemsize * array[0].size
        chunk_size = self._nrows_in_chunks * size_per_row

        if max_cached_chunks is None:
            buffer_size = max(buffer_size, chunk_size)
            chunks_in_cache = buffer_size // chunk_size
        else:
            chunks_in_cache = max_cached_chunks

        LOG.info(
            f"Initializing ReadAheadWriteBehindBuffer with chunk shape {array.chunks}, "
            f"caching {chunks_in_cache} chunks ({chunks_in_cache * chunk_size / 1024 / 1024:.2f} MB)",
        )

        self._lru_chunks_cache = LRU(chunks_in_cache, callback=self._evict_buffer)
        self._lock = threading.RLock()
        self.chunks_in_cache = chunks_in_cache

    @property
    def max_cached_chunks(self) -> int:
        """The maximum number of cached chunks."""
        return self.chunks_in_cache

    @staticmethod
    def _evict_buffer(key: int, buffer: _Buffer) -> None:
        """Callback called when a buffer is evicted from the cache.

        Parameters
        ----------
        key : int
            The buffer index.
        buffer : _Buffer
            The buffer to evict.
        """
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Evicting buffer {key} {buffer}")

        buffer.flush()

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
            key = self._normalise_key(key)
            chunk = self._get_key_chunk(key)
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
            key = self._normalise_key(key)
            chunk = self._get_key_chunk(key)
            return chunk[key]

    def flush(self) -> None:
        """Flush all cached buffers to the underlying Zarr array."""
        with self._lock:
            for buffer in self._lru_chunks_cache.values():
                buffer.flush()

    @property
    def _max_chunk_index(self) -> int:
        """The maximum chunk index based on the array size."""
        return (self._arr.shape[0] - 1) // self._nrows_in_chunks

    def _ensure_chunk_in_cache(self, chunk_index: int) -> _Buffer:
        """Ensure a chunk is present in the cache.

        Parameters
        ----------
        chunk_index : int
            The index of the chunk.

        Returns
        -------
        _Buffer
            The cached chunk.
        """
        chunk_index = int(chunk_index)  # Ensure chunk_index is an int, in case it was passed as a numpy integer
        with self._lock:
            if chunk_index in self._lru_chunks_cache:
                return self._lru_chunks_cache[chunk_index]

        # Create the chunk outside the lock so we don't block other operations while loading from disk
        # In the unlikely event of multiple threads loading the same chunk, the LRU cache will handle duplicates
        # and the garbage collector will clean up unreferenced chunks
        chunk = _Buffer(self._arr, chunk_index)

        with self._lock:

            # For debugging purposes, check again if the chunk was loaded while we were outside the lock
            if self._no_reload:
                if chunk_index in self._was_loaded:
                    raise RuntimeError(f"Chunk {chunk_index} has already been loaded once, re-loading is disabled")
                self._was_loaded.add(chunk_index)

            self._lru_chunks_cache[chunk_index] = chunk

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(
                    f"Loaded chunk {chunk_index} {self._lru_chunks_cache[chunk_index]}. Cached: {sorted(self._lru_chunks_cache.keys())},"
                    f" {sum(chunk.cache.nbytes for chunk in self._lru_chunks_cache.values()) / 1024 / 1024:.2f} MB"
                    f" dirty={sum(chunk.dirty for chunk in self._lru_chunks_cache.values())}",
                )

            return self._lru_chunks_cache[chunk_index]

    def _normalise_key(self, key: Any) -> Any:
        """Normalises the key to absolute indices (no Nones, no negatives).

        Parameters
        ----------
        key : Any
            The key to normalise.

        Returns
        -------
        Any
            The normalised key.
        """
        match key:
            case int():
                if key < 0:
                    key += self._arr.shape[0]
                return key

            case slice():
                result = slice(*key.indices(self._arr.shape[0]))
                assert result.step >= 1, "Negative or zero step slices are not supported"
                assert result.start <= result.stop, "Slice start must be less than or equal to stop"
                assert 0 <= result.start, "Slice start negative after normalisation"
                return result

            case tuple():
                return (self._normalise_key(key[0]),) + key[1:]

            case list():
                return self._normalise_key(np.array(key))

            case np.ndarray():
                assert key.ndim == 1, "Only 1D np.ndarray are supported"

                if key.dtype == bool:
                    return np.flatnonzero(key)

                if np.issubdtype(key.dtype, np.integer):
                    if np.any(key < 0):
                        key = np.where(key < 0, key + self._arr.shape[0], key)
                    return key

                raise TypeError(f"Unsupported np.ndarray dtype: {key.dtype}")

            case np.integer():
                return self._normalise_key(int(key))

            case _:
                raise TypeError(f"Unsupported key type: {type(key)} ({key})")

    def _get_key_chunk(self, key: Any) -> _Buffer:

        match key:

            case int():
                chunk_index = key // self._nrows_in_chunks
                return self._ensure_chunk_in_cache(chunk_index)

            case tuple():
                return self._get_key_chunk(key[0])

            case slice():

                start, stop, step = key.start, key.stop, key.step

                if start == stop:
                    # Handle empty slice
                    in_lru = list(self._lru_chunks_cache.keys())
                    chunk_index = in_lru[0] if in_lru else 0
                    return self._ensure_chunk_in_cache(chunk_index)

                last = start + step * ((stop - 1 - start) // step)
                start_chunk = start // self._nrows_in_chunks
                last_chunk = last // self._nrows_in_chunks

                if start_chunk == last_chunk:
                    # The index is within a single chunk
                    return self._ensure_chunk_in_cache(start_chunk)

                # The index spans multiple chunks, use multi-chunk handler
                return _MultiBufferSpan(self, self._arr.shape[0])

            case np.ndarray():

                if np.any(key < 0):
                    key = np.where(key < 0, key + self._arr.shape[0], key)
                unique_chunks = np.unique(key // self._nrows_in_chunks)
                if len(unique_chunks) == 1:
                    return self._ensure_chunk_in_cache(int(unique_chunks[0]))

                return _MultiBufferSpan(self, self._arr.shape[0])

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

            assert self._read_ahead is None, "Resizing not supported with read-ahead enabled"

            self._arr.resize(*new_shape)

            for chunk in list(self._lru_chunks_cache.values()):
                chunk.resize(self._lru_chunks_cache, new_shape)

            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(f"Resized underlying array from {self._arr.shape} to {new_shape}")

            self._nrows_in_chunks = self._arr.chunks[0]

    def __enter__(self) -> "ReadAheadWriteBehindBuffer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()


class _MultiBufferSpan:
    """Handles multi-chunk slicing for ReadAheadWriteBehindBuffer."""

    def __init__(self, cache: ReadAheadWriteBehindBuffer, array_size: int):
        """Initialise a multi-chunk slice handler.

        Parameters
        ----------
        cache : ReadAheadWriteBehindBuffer
            The chunk cache.
        """
        self._cache = cache
        self._array_size = array_size

    def __setitem__(self, key: tuple[Any, ...], value: Any) -> None:
        """Set values across multiple chunks.

        Parameters
        ----------
        key : tuple
            The key representing the slice.
        value : Any
            The value to set.
        """
        for chunk, idx, value_key in self._split_chunks(key):
            chunk[idx] = value[value_key]

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
        for chunk, idx, value_key in self._split_chunks(key):
            values.append(chunk[idx])

        return np.concatenate(values, axis=0)

    def _split_chunks(
        self, key: Any, remaining_key: tuple[Any, ...] = ()
    ) -> Iterator[tuple[Any, list[int], list[int]]]:
        """Splits a list of global indices into per-chunk local indices."""

        match key:
            case tuple():
                yield from self._split_chunks(key[0], key[1:])
                return

            case slice():
                yield from self._split_slice_chunks(key, remaining_key)
                return

            case np.ndarray() | list():
                yield from self._split_array_chunks(key, remaining_key)
                return

        raise TypeError(f"Unsupported key type for multi-chunk slicing: {type(key)}")

    def _split_slice_chunks(self, key: slice, remaining_key: tuple[Any, ...]) -> Iterator[tuple[Any, list[int], slice]]:
        """Faster version of _split_array_chunks for slice keys."""

        start, stop, step = key.indices(self._array_size)
        if start >= stop:
            return

        chunk_size = self._cache._nrows_in_chunks
        idx = start
        offset = 0

        while idx < stop:
            chunk_number = idx // chunk_size
            # The first index of the NEXT chunk
            next_boundary = (chunk_number + 1) * chunk_size

            # Calculate how many steps we can take before hitting or crossing the boundary
            # Formula: ceil((boundary - current) / step)
            steps_to_boundary = (next_boundary - idx + step - 1) // step

            # Ensure we don't go past the slice's stop point
            steps_to_stop = (stop - idx + step - 1) // step
            actual_steps = min(steps_to_boundary, steps_to_stop)

            segment_stop = idx + (actual_steps * step)

            indices = np.arange(idx, segment_stop, step)

            yield (
                self._cache._ensure_chunk_in_cache(chunk_number),
                (indices,) + remaining_key,
                slice(offset, offset + len(indices)),
            )

            offset += len(indices)
            idx = segment_stop

    def _split_array_chunks(
        self, key: np.ndarray, remaining_key: tuple[Any, ...]
    ) -> Iterator[tuple[Any, list[int], slice]]:

        # Use Numpy for speed
        key = np.asanyarray(key)

        if key.size == 0:
            return

        chunk_size = self._cache._nrows_in_chunks
        chunk_ids = key // chunk_size

        # Find where the chunk ID changes
        change_points = np.flatnonzero(chunk_ids[1:] != chunk_ids[:-1]) + 1

        # Split indices and chunk IDs at the change points
        split_indices = np.split(key, change_points)
        split_chunks = np.split(chunk_ids, change_points)

        offset = 0
        for i in range(len(split_indices)):
            chunk_number = int(split_chunks[i][0])
            indices = split_indices[i]
            yield (
                self._cache._ensure_chunk_in_cache(chunk_number),
                (indices,) + remaining_key,
                slice(offset, offset + len(indices)),
            )
            offset += len(indices)


class ReadAheadBuffer(ReadAheadWriteBehindBuffer):
    """Caches chunks of a Zarr array for efficient read access with read-ahead."""

    def __init__(self, array, start=0, *args, **kwargs: Any):

        super().__init__(array, *args, **kwargs, no_reload=True)
        self._read_ahead = ThreadPoolExecutor(max_workers=1)
        self._read_ahead.submit(self._read_ahead_worker, slice(start, self.chunks_in_cache + start, 1))

    def __setitem__(self, key, value):
        raise RuntimeError("ReadAheadBuffer is read-only")

    def __getitem__(self, key):
        key = self._normalise_key(key)

        if isinstance(key, tuple):
            first_key = key[0]
        else:
            first_key = key

        match first_key:
            case int():
                chunk_index = first_key // self._nrows_in_chunks
                self._read_ahead.submit(self._read_ahead_worker, slice(chunk_index + 1, chunk_index + 2, 1))

            case slice():
                # TODO: optimize for step > 1
                start, stop, _ = first_key.indices(self._arr.shape[0])
                start = start // self._nrows_in_chunks
                stop = (stop - 1) // self._nrows_in_chunks + 1
                self._read_ahead.submit(self._read_ahead_worker, slice(start, stop, 1))
            case _:
                pass

        return super().__getitem__(key)

    def _read_ahead_worker(self, index: slice) -> None:
        """Worker function for read-ahead, loads the next buffer in the background.

        Parameters
        ----------
        index : slice
            The slice of buffer indices to read ahead.
        """
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Read-ahead loading buffer {index}")

        for i in range(index.start, index.stop, index.step):
            if i > self._max_buffer_index:
                break

            self._ensure_chunk_in_cache(i)


class RandomReadBuffer(ReadAheadWriteBehindBuffer):
    """Caches buffers of a Zarr array for efficient read access with read-ahead."""

    def __init__(self, *args, **kwargs: Any):

        super().__init__(
            **kwargs,
            read_ahead=False,
        )

    def __setitem__(self, key, value):
        raise RuntimeError("RandomReadBuffer is read-only")


class WriteBehindBuffer(ReadAheadWriteBehindBuffer):
    pass
