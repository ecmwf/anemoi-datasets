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

from anemoi.datasets.caching import ChunksCache

SIZE = 1_000_000
SHAPE = (SIZE, 10)


def make_array():

    data = np.zeros(SHAPE, dtype=np.int32)
    for i in range(SIZE):
        data[i] = np.arange(i, i + 10, dtype=np.int32)

    random = np.random.randint(-SIZE, SIZE, size=SHAPE, dtype=np.int32)

    array = zarr.open_array(
        store=zarr.MemoryStore(),
        shape=data.shape,
        chunks=(1000, 10),
        dtype=data.dtype,
        mode="w",
    )
    array[:] = data

    return array, data, random


def _test_keys(key):
    return [key, (key, slice(None)), (key, 2), (key, slice(0, 5, 2))]


def _test_read(array, data, random, key):

    for test_key in _test_keys(key):
        try:
            assert np.array_equal(array[test_key], data[test_key]), f"Data mismatch for key {test_key}"
        except Exception as e:
            logging.error(f"Error reading key {test_key}: {e}")
            raise


def _test_write(array, data, random, key):

    for test_key in _test_keys(key):
        array[test_key] = random[test_key]
        data[test_key] = random[test_key]

    _test_read(array, data, random, key)


def _test_cache(test_func):
    array, data, random = make_array()
    array = ChunksCache(array, max_cached_chunks=10)
    chunk_size = array.chunks[0]

    test_func(array, data, random, 0)
    test_func(array, data, random, -1)

    test_func(array, data, random, slice(None))

    for i in range(0, SIZE, chunk_size):
        test_func(array, data, random, i)

    for i in range(0, SIZE, chunk_size + 10):
        test_func(array, data, random, i)

    for i in range(0, SIZE, chunk_size - 10):
        test_func(array, data, random, i)

    for i in range(0, SIZE, chunk_size):
        test_func(array, data, random, slice(i, i + 5))

    for i in range(0, SIZE, chunk_size):
        test_func(array, data, random, slice(i, i + chunk_size // 2))

    for i in range(0, SIZE, chunk_size):
        test_func(array, data, random, slice(i, i + chunk_size * 2))

    test_func(array, data, random, np.array([0, 10, 20, 30, 40]))
    test_func(array, data, random, np.array([SIZE - 1, SIZE - 10, SIZE - 20]))
    test_func(array, data, random, np.array([True if i % 2 == 0 else False for i in range(SIZE)]))
    test_func(array, data, random, [i * 1000 for i in range(SIZE // 1000)])


def test_cache_read():
    _test_cache(_test_read)


def test_cache_write():

    _test_cache(_test_write)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
