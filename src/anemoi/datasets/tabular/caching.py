# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import zarr
from lru import LRU

LOG = logging.getLogger(__name__)


class _Chunk:

    def __init__(self, array: zarr.Array, chunk_index: int):
        self.array = array
        self.chunk_index = chunk_index
        self.chunk_size = array.chunks[0]
        self.offset = chunk_index * self.chunk_size
        self.cache = array[self.offset : min(self.offset + self.chunk_size, array.shape[0])]
        self.dirty = False
        self.loaded = False

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        return f"<Chunk[offset={self.offset} shape={self.cache.shape} dirty={self.dirty}]>"

    def __setitem__(self, key, value):
        local_index = self._local_index(key)
        self.cache[local_index] = value
        self.dirty = True

    def __getitem__(self, key):
        local_index = self._local_index(key)
        return self.cache[local_index]

    def _local_index(self, key):
        """Checks and converts the global key to a local index within the chunk."""

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

    def flush(self):
        """Flush cached changes to Zarr array"""
        if self.dirty:
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(f"Flushing chunk {self.chunk_index} {self}")
            self.array[self.offset : self.offset + len(self.cache)] = self.cache
            self.dirty = False

    def _remove(self, lru, why):
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Removing chunk {self.chunk_index} {self} because {why}")
        self.flush()
        del lru[self.chunk_index]

    def resize(self, lru, new_array_shape):
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
    def __init__(self, array: zarr.Array, chunk_caching=512 * 1024 * 1024):
        self._arr = array
        self._chunk_0 = array.chunks[0]

        size_per_row = np.dtype(array.dtype).itemsize * array[0].size
        chunk_size = self._chunk_0 * size_per_row
        chunk_caching = max(chunk_caching, chunk_size)
        chunks_in_cache = max(chunk_caching // chunk_size, 1)

        LOG.info(
            f"Initializing ChunksCache with chunk size {array.chunks}, "
            f"caching {chunks_in_cache} chunks ({chunks_in_cache * chunk_size / 1024 / 1024:.2f} MB)",
        )

        self._chunks = LRU(chunks_in_cache, callback=self._evict_chunk)

    def _evict_chunk(self, key, chunk: _Chunk):
        # print(f"Evicting chunk {key} {chunk}", flush=True)
        chunk.flush()

    def __setitem__(self, key, value):
        chunk, key = self._get_chunk(key)
        chunk[key] = value

    def __getitem__(self, key):
        chunk, key = self._get_chunk(key)
        return chunk[key]

    def flush(self, reset=False):
        for chunk in self._chunks.values():
            chunk.flush()

        if reset:
            self._chunks.clear()

    def _ensure_chunk(self, chunk_index):
        if chunk_index not in self._chunks:
            self._chunks[chunk_index] = _Chunk(self._arr, chunk_index)
            if LOG.isEnabledFor(logging.DEBUG):
                LOG.debug(
                    f"Loaded chunk {chunk_index} {self._chunks[chunk_index]}. Cached: {sorted(self._chunks.keys())},"
                    f" {sum(chunk.cache.nbytes for chunk in self._chunks.values()) / 1024 / 1024:.2f} MB"
                    f" dirty={sum(chunk.dirty for chunk in self._chunks.values())}",
                )
        return self._chunks[chunk_index]

    def _normalise_slice(self, key):
        # Convert to absolute indices (no Nones, no negatives)
        return slice(*key.indices(self._arr.shape[0]))

    def _get_chunk(self, key):
        """Retrieves the appropriate chunk and a normalised key."""
        match key:

            case int():
                chunk_index = key // self._chunk_0
                return self._ensure_chunk(chunk_index), key

            case slice():
                return self._get_chunk(key, slice(None, None, None))

            case tuple():
                if isinstance(key[0], int):
                    chunk_index = key[0] // self._chunk_0
                    return self._ensure_chunk(chunk_index), key

                if isinstance(key[0], slice):
                    key = (self._normalise_slice(key[0]),) + key[1:]
                    indices = set(a // self._chunk_0 for a in range(*key[0].indices(self._arr.shape[0])))
                    if len(indices) == 1:
                        chunk_index = indices.pop()
                        return self._ensure_chunk(chunk_index), key
                    else:
                        return _MultiChunkSlice(self, key), key

                raise TypeError(f"Unsupported key type in tuple: {type(key[0])} ({key[0]})")

            case _:
                raise TypeError(f"Unsupported key type: {type(key)} ({key})")

    @property
    def attrs(self):
        return self._arr.attrs

    @property
    def shape(self):
        return self._arr.shape

    @property
    def chunks(self):
        return self._arr.chunks

    def __len__(self):
        return len(self._arr)

    def resize(self, *new_shape):
        for chunk in self._chunks.values():
            chunk.resize(self._chunks, new_shape)
        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug(f"Resized underlying array from {self._arr.shape} to {new_shape}")
        self._arr.resize(*new_shape)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()


class _MultiChunkSlice:
    def __init__(self, cache: ChunksCache, key):
        self._cache = cache
        self._key = key
        assert isinstance(key, tuple) and isinstance(key[0], slice)

    def __setitem__(self, key, value):
        for chunk, idx, shape in self._split_chunks(key):
            chunk[idx] = value[shape]

    def __getitem__(self, key):
        values = []
        for chunk, idx, shape in self._split_chunks(key):
            values.append(chunk[idx][shape])
        return np.concatenate(values, axis=0)

    def _split_chunks(self, key):
        assert self._key == key
        slice0 = key[0]
        # Assume slice is normalised
        start, stop, step = slice0.start, slice0.stop, slice0.step

        bits = {}
        current = start

        while current < stop:
            chunk_index = current // self._cache._chunk_0
            if chunk_index not in bits:
                bits[chunk_index] = slice(current, current + 1, step)
            else:
                bits[chunk_index] = slice(bits[chunk_index].start, current + 1, step)
            current += step

        shape = 0
        for chunk_index, s in sorted(bits.items()):
            chunk = self._cache._ensure_chunk(chunk_index)
            end = min(len(chunk), self._cache._chunk_0, (s.stop - s.start) // s.step)
            yield chunk, (s,) + key[1:], slice(shape, shape + end)
            shape += end
