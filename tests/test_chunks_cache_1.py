import numpy as np
import zarr

from anemoi.datasets.buffering import ChunksCache


def create_zarr_array(shape=(20, 3), chunks=(5, 3), dtype=np.int64):
    store = zarr.MemoryStore()
    arr = zarr.create(shape=shape, chunks=chunks, dtype=dtype, store=store, overwrite=True)
    arr[:] = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    return arr


def test_chunks_cache_get_set():
    arr = create_zarr_array()
    cache = ChunksCache(arr, max_cached_chunks=6)
    # Test get
    assert np.all(cache[0] == arr[0])
    # Test set
    cache[0] = np.array([100, 101, 102])
    assert np.all(cache[0] == np.array([100, 101, 102]))
    cache.flush()
    assert np.all(arr[0] == np.array([100, 101, 102]))


def test_chunks_cache_slice():
    arr = create_zarr_array()
    cache = ChunksCache(arr, max_cached_chunks=6)
    # Test slice get
    assert np.all(cache[5:10] == arr[5:10])
    # Test slice set
    cache[5:10] = np.ones((5, 3), dtype=np.int64)
    assert np.all(cache[5:10] == np.ones((5, 3), dtype=np.int64))
    cache.flush()
    assert np.all(arr[5:10] == np.ones((5, 3), dtype=np.int64))


def test_chunks_cache_array_index():
    arr = create_zarr_array()
    cache = ChunksCache(arr, max_cached_chunks=6)
    idx = np.array([1, 3, 5, 7])
    assert np.all(cache[idx] == arr[idx])
    cache[idx] = np.full((4, 3), 7, dtype=np.int64)
    assert np.all(cache[idx] == 7)
    cache.flush()
    assert np.all(arr[idx] == 7)


def test_chunks_cache_len_and_resize():
    arr = create_zarr_array()
    cache = ChunksCache(arr, max_cached_chunks=6)
    assert len(cache) == 20
    cache[:]
    cache.resize(25, 3)
    assert cache.shape == (25, 3)
    assert arr.shape == (25, 3)


def test_chunks_cache_context_manager():
    arr = create_zarr_array()
    with ChunksCache(arr, max_cached_chunks=6) as cache:
        cache[0] = np.array([123, 456, 789])


def test_chunks_cache_resize_larger_first_dim():
    arr = create_zarr_array(shape=(10, 3))
    cache = ChunksCache(arr, max_cached_chunks=6)
    cache[:]
    cache.resize(15, 3)
    assert cache.shape == (15, 3)
    assert cache[:].shape == (15, 3)
    assert arr.shape == (15, 3)
    # New rows should be zero-initialized
    assert np.all(arr[10:] == 0)
    assert np.all(cache[10:] == 0)


def test_chunks_cache_resize_smaller_first_dim():
    arr = create_zarr_array(shape=(10, 3))
    cache = ChunksCache(arr, max_cached_chunks=6)
    cache[:]
    cache.resize(5, 3)
    assert cache.shape == (5, 3)
    assert cache[:].shape == (5, 3)
    assert arr.shape == (5, 3)
    # Data should match original for remaining rows
    assert np.all(cache[:] == np.arange(15).reshape(5, 3))


def test_chunks_cache_resize_larger_second_dim():
    arr = create_zarr_array(shape=(5, 2))
    cache = ChunksCache(arr, max_cached_chunks=6)
    cache[:]
    cache.resize(5, 4)
    assert cache.shape == (5, 4)
    assert cache[:].shape == (5, 4)
    assert arr.shape == (5, 4)
    # New columns should be zero-initialized
    assert np.all(arr[:, 2:] == 0)
    assert np.all(cache[:, 2:] == 0)


def test_chunks_cache_resize_smaller_second_dim():
    arr = create_zarr_array(shape=(5, 4))
    save = arr[:].copy()
    cache = ChunksCache(arr, max_cached_chunks=6)
    cache[:]
    cache.resize(5, 2)
    assert cache.shape == (5, 2)
    assert cache[:].shape == (5, 2)
    assert arr.shape == (5, 2)
    # Data should match original for remaining columns
    assert np.all(cache[:] == save[:, :2])


def test_chunks_cache_resize_both_dims():
    arr = create_zarr_array(shape=(6, 6))
    cache = ChunksCache(arr, max_cached_chunks=6)
    cache[:]
    cache.resize(8, 4)
    assert cache.shape == (8, 4)
    assert cache[:].shape == (8, 4)
    assert arr.shape == (8, 4)
    # New rows and columns should be zero-initialized
    assert np.all(arr[6:, :] == 0)
    assert np.all(arr[:, 4:] == 0) if arr.shape[1] > 4 else True


def test_chunks_cache_resize_and_set():
    arr = create_zarr_array(shape=(4, 4))
    cache = ChunksCache(arr, max_cached_chunks=6)
    cache[:]
    cache.resize(6, 4)
    assert cache.shape == (6, 4)
    cache[4:] = np.ones((2, 4), dtype=np.int64) * 9
    assert np.all(cache[4:] == 9)
    cache.flush()
    assert np.all(arr[4:] == 9)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
