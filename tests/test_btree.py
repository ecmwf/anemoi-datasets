import shutil
import tempfile

import numpy as np

from anemoi.datasets.date_indexing import btree


def make_temp_zarr():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_btree_set_get():
    with tempfile.TemporaryDirectory() as temp_dir:
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="w", page_size=4)
        # Insert and retrieve
        z.set(100, (1, 2))
        z.set(200, (3, 4))
        assert z.get(100) == (1, 2)
        assert z.get(200) == (3, 4)
        assert z.get(300) is None


def test_btree_overwrite():
    with tempfile.TemporaryDirectory() as temp_dir:
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="w", page_size=4)
        z.set(100, (1, 2))
        z.set(100, (5, 6))
        assert z.get(100) == (5, 6)


def test_btree_bulk_load_1():
    with tempfile.TemporaryDirectory() as temp_dir:
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="w", page_size=4)
        data = np.array([[i, i * 2, i * 3] for i in range(20)], dtype=np.int64)
        z.bulk_load(data)
        z.dump()
        print(z.size())
        print(z.count())
        for i in range(20):
            assert z.get(i) == (i * 2, i * 3)


def test_btree_bulk_load_2():
    with tempfile.TemporaryDirectory() as temp_dir:
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="w", page_size=100)
        data = np.array([[i, i * 2, i * 3] for i in range(20)], dtype=np.int64)
        z.bulk_load(data)
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="r", page_size=100)
        z.dump()
        print(z.size())
        print(z.count())
        for i in range(20):
            assert z.get(i) == (i * 2, i * 3)


def test_btree_range():
    with tempfile.TemporaryDirectory() as temp_dir:
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="w", page_size=4)
        data = np.array([[i, i, i] for i in range(10)], dtype=np.int64)
        z.bulk_load(data)
        results = z.range(3, 7)
        keys = [k for k, v in results]
        assert keys == list(range(3, 8))


def test_btree_first_last_keys():
    with tempfile.TemporaryDirectory() as temp_dir:
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="w", page_size=4)
        data = np.array([[i, i, i] for i in range(10)], dtype=np.int64)
        z.bulk_load(data)
        first, last = z.first_last_keys()
        assert first == 0
        assert last == 9


def test_btree_boundaries():
    with tempfile.TemporaryDirectory() as temp_dir:
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="w", page_size=4)
        data = np.array([[i, i, i] for i in range(10)], dtype=np.int64)
        z.bulk_load(data)
        first, last = z.boundaries(2, 6)
        assert first[0] == 2
        assert last[0] == 6


def test_btree_size_and_count():
    with tempfile.TemporaryDirectory() as temp_dir:
        z = btree.ZarrBTree(path=temp_dir, name="test_btree", mode="w", page_size=4)
        data = np.array([[i, 1, 1] for i in range(10)], dtype=np.int64)
        z.bulk_load(data)
        assert z.size() == 10
        assert z.count() == 10
