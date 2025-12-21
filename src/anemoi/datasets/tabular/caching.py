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
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Iterator
from typing import Tuple

import numpy as np
import zarr
from lru import LRU

LOG = logging.getLogger(__name__)


class _Chunk:
    """Represents a chunk of a Zarr array, providing caching and synchronisation.
    Only the first dimension is considered for caching. Other chuncking
    dimensions cached together with the first dimension.

    Parameters
    ----------
    array : zarr.Array
        The Zarr array to cache chunks from.
    chunk_index : int
        The index of the chunk within the array.
    """

    def __init__(self, array: zarr.Array, chunk_index: int):
        self.array = array
        self.chunk_index = chunk_index
        self.chunk_size = array.chunks[0]
        self.offset = chunk_index * self.chunk_size
        # use min to avoid indexing out of bounds of array
        self.cache = array[self.offset : min(self.offset + self.chunk_size, array.shape[0])]
        self.dirty = False
        self.lock = threading.RLock()

    def __len__(self) -> int:
        with self.lock:
            return len(self.cache)

    def __repr__(self) -> str:
        return f"<Chunk[offset={self.offset} shape={self.cache.shape} dirty={self.dirty}]>"

    def __setitem__(self, key: Any, value: Any) -> None:
        with self.lock:
            local_index = self._local_index(key)
            self.cache[local_index] = value
            self.dirty = True

    def __getitem__(self, key: Any) -> Any:
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

    def resize(self, lru: LRU, new_array_shape: Tuple[int, ...]) -> None:
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
    """Caches chunks of a Zarr array for efficient access and modification.

    Parameters
    ----------
    array : zarr.Array
        The Zarr array to cache.
    chunk_caching : int, optional
        The cache size in bytes (default is 512 * 1024 * 1024).
    read_ahead : bool, optional
        Whether to enable read-ahead (default is False).
    """

    def __init__(self, array: zarr.Array, chunk_caching: int = 512 * 1024 * 1024, read_ahead: bool = False):
        self._arr = array
        self._nrows_in_chunks = array.chunks[0]

        size_per_row = np.dtype(array.dtype).itemsize * array[0].size
        chunk_size = self._nrows_in_chunks * size_per_row
        chunk_caching = max(chunk_caching, chunk_size)
        chunks_in_cache = max(chunk_caching // chunk_size, 1)

        LOG.info(
            f"Initializing ChunksCache with chunk shape {array.chunks}, "
            f"caching {chunks_in_cache} chunks ({chunks_in_cache * chunk_size / 1024 / 1024:.2f} MB)",
        )

        self._chunks_in_cache = chunks_in_cache
        self._chunks = LRU(chunks_in_cache, callback=self._evict_chunk)
        self._lock = threading.RLock()

        self._last_read_ahead_chunk = 0

        if read_ahead:
            self._read_ahead = ThreadPoolExecutor(max_workers=1)
            self._last_read_ahead_chunk = 0
            self._read_ahead.submit(self._read_ahead_worker, 0)
        else:
            self._read_ahead = None

    def _read_ahead_worker(self, chunk_number: int) -> None:
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Read-ahead loading chunk {chunk_number}")
            if chunk_number > self._max_chunk_index:
                return
        self._ensure_chunk_in_cache(chunk_number, from_read_ahead=True)

    @staticmethod
    def _evict_chunk(key: int, chunk: _Chunk) -> None:
        """Evicts a chunk from the cache, flushing it if dirty.

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
        with self._lock:
            chunk, key = self._get_key_chunk(key)
            chunk[key] = value

    def __getitem__(self, key: Any) -> Any:

        with self._lock:
            chunk, key = self._get_key_chunk(key)
            return chunk[key]

    def flush(self) -> None:
        """Flush all cached chunks to the underlying Zarr array."""
        with self._lock:
            for chunk in self._chunks.values():
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
            if chunk_index in self._chunks:
                return self._chunks[chunk_index]

        # Create the chunk outside the lock for speed
        chunk = _Chunk(self._arr, chunk_index)

        with self._lock:
            self._chunks[chunk_index] = chunk

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(
                f"Loaded chunk {chunk_index} {self._chunks[chunk_index]}. Cached: {sorted(self._chunks.keys())},"
                f" {sum(chunk.cache.nbytes for chunk in self._chunks.values()) / 1024 / 1024:.2f} MB"
                f" dirty={sum(chunk.dirty for chunk in self._chunks.values())}",
            )

        if self._read_ahead is not None and not from_read_ahead:
            with self._lock:
                if chunk_index + 1 > self._last_read_ahead_chunk:
                    self._last_read_ahead_chunk = chunk_index + 1
                    self._read_ahead.submit(self._read_ahead_worker, chunk_index + 1)

        with self._lock:
            return self._chunks[chunk_index]

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
        return slice(*key.indices(self._arr.shape[0]))

    def _get_key_chunk(self, key: Any) -> Tuple[Any, Any]:
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
                chunk_index = key // self._nrows_in_chunks
                return self._ensure_chunk_in_cache(chunk_index), key

            case slice():
                return self._get_key_chunk((key, slice(None, None, None)))

            case tuple():
                if isinstance(key[0], int):
                    chunk_index = key[0] // self._nrows_in_chunks
                    return self._ensure_chunk_in_cache(chunk_index), key

                if isinstance(key[0], slice):
                    key = (self._normalise_slice(key[0]),) + key[1:]
                    indices = set(a // self._nrows_in_chunks for a in range(*key[0].indices(self._arr.shape[0])))
                    if len(indices) == 1:
                        chunk_index = indices.pop()
                        return self._ensure_chunk_in_cache(chunk_index), key
                    else:
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
    def chunks(self):
        """The chunk shape of the underlying Zarr array."""
        return self._arr.chunks

    def __len__(self) -> int:
        return len(self._arr)

    def resize(self, *new_shape: int) -> None:
        """Resize the underlying Zarr array and adjust cached chunks.

        Parameters
        ----------
        *new_shape : int
            The new shape of the array.
        """
        with self._lock:
            for chunk in self._chunks.values():
                chunk.resize(self._chunks, new_shape)
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(f"Resized underlying array from {self._arr.shape} to {new_shape}")
            self._arr.resize(*new_shape)

    def __enter__(self) -> "ChunksCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()


class _MultiChunkSlice:
    """Handles multi-chunk slicing for ChunksCache.

    Parameters
    ----------
    cache : ChunksCache
        The chunk cache.
    key : tuple
        The key representing the slice.
    """

    def __init__(self, cache: ChunksCache, key: Tuple[Any, ...]):
        self._cache = cache
        self._key = key
        assert isinstance(key, tuple) and isinstance(key[0], slice)

    def __setitem__(self, key: Tuple[Any, ...], value: Any) -> None:
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

    def __getitem__(self, key: Tuple[Any, ...]) -> Any:
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

    def _split_chunks(self, key: Tuple[Any, ...]) -> Iterator[Tuple[_Chunk, Tuple[Any, ...], slice]]:
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

        bits = {}
        current = start

        while current < stop:
            chunk_index = current // self._cache._nrows_in_chunks
            if chunk_index not in bits:
                bits[chunk_index] = slice(current, current + 1, step)
            else:
                bits[chunk_index] = slice(bits[chunk_index].start, current + 1, step)
            current += step

        shape = 0
        for chunk_index, s in sorted(bits.items()):
            chunk = self._cache._ensure_chunk_in_cache(chunk_index)
            end = min(len(chunk), self._cache._nrows_in_chunks, (s.stop - s.start) // s.step)
            yield chunk, (s,) + key[1:], slice(shape, shape + end)
            shape += end
