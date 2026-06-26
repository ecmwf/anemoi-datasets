"""Tests for tabular dataset sharding (open_dataset(..., sharding=N))."""

import numpy as np
import pytest

# Reuse the synthetic tabular store builder from the window-view tests.
from test_window_view import _create_tabular_store

from anemoi.datasets.usage.tabular.sharding import ShardedTabular
from anemoi.datasets.usage.tabular.sharding import shard_dataset
from anemoi.datasets.usage.tabular.sharding import single_shard
from anemoi.datasets.usage.tabular.store import TabularZarr
from anemoi.datasets.windows.view import WindowView

COUNT = 8


@pytest.fixture(scope="module")
def tabular_store():
    root, _events = _create_tabular_store("btree")
    # Attributes needed to open the store as a tabular dataset.
    root.attrs["layout"] = "tabular"
    root.attrs["variables"] = ["v0", "v1", "v2"]
    return root


def _full_window(view, index):
    return np.asarray(view[index])


def test_window_view_shards_partition_rows(tabular_store):
    """Concatenating all shards of a window reproduces the full window exactly."""
    view = WindowView(tabular_store)
    shards = [view._internal_set_shard(i, COUNT) for i in range(COUNT)]

    # Test a spread of window indices, including some likely non-empty ones.
    indices = list(range(0, len(view), max(1, len(view) // 20)))

    seen_non_empty = False
    for index in indices:
        full = _full_window(view, index)
        pieces = [np.asarray(s[index]) for s in shards]

        # Row counts add up and each piece is at most one row larger than 1/N.
        lengths = [p.shape[0] for p in pieces]
        assert sum(lengths) == full.shape[0], (index, lengths, full.shape)
        assert max(lengths) - min(lengths) <= 1, (index, lengths)

        # Exact reconstruction of the data.
        recombined = np.concatenate(pieces, axis=0) if pieces else full
        np.testing.assert_array_equal(recombined, full)

        if full.shape[0] > 0:
            seen_non_empty = True

    assert seen_non_empty, "test did not exercise any non-empty window"


def test_window_view_shard_metadata_aligned(tabular_store):
    """Per-row metadata (dates/lat/lon) stays aligned with the sharded rows."""
    view = WindowView(tabular_store)
    shards = [view._internal_set_shard(i, COUNT) for i in range(COUNT)]

    # Pick the window with the most rows for a meaningful check.
    index = max(range(len(view)), key=lambda i: view[i].shape[0])

    full = view[index]
    pieces = [s[index] for s in shards]

    np.testing.assert_array_equal(np.concatenate([p.dates for p in pieces]), full.dates)
    np.testing.assert_array_equal(np.concatenate([p.latitudes for p in pieces]), full.latitudes)
    np.testing.assert_array_equal(np.concatenate([p.longitudes for p in pieces]), full.longitudes)

    # Boundaries of a single shard cover that shard's rows.
    for piece in pieces:
        (boundary,) = piece.boundaries
        assert boundary == slice(0, piece.shape[0])


def test_window_view_shard_chaining_raises(tabular_store):
    view = WindowView(tabular_store)._internal_set_shard(0, COUNT)
    with pytest.raises(ValueError, match="cannot be chained"):
        view._internal_set_shard(0, 2)


@pytest.mark.parametrize("index,count", [(-1, 4), (4, 4), (0, 0)])
def test_window_view_shard_bad_args(tabular_store, index, count):
    with pytest.raises(ValueError):
        WindowView(tabular_store)._internal_set_shard(index, count)


def test_shard_dataset_returns_list_of_shards(tabular_store):
    ds = TabularZarr(tabular_store)
    shards = shard_dataset(ds, COUNT)

    assert len(shards) == COUNT
    assert all(isinstance(s, TabularZarr) for s in shards)

    # Window count, dates and variables are identical across shards.
    for s in shards:
        assert len(s) == len(ds)
        np.testing.assert_array_equal(s.dates, ds.dates)
        assert s.variables == ds.variables

    # Original dataset is untouched (cloning, not mutation).
    assert ds._window_view.shard is None

    # Rows partition the full dataset for a non-empty window.
    index = max(range(len(ds)), key=lambda i: ds[i].shape[0])
    full = ds[index]
    pieces = [s[index] for s in shards]
    assert sum(p.shape[0] for p in pieces) == full.shape[0]
    np.testing.assert_array_equal(np.concatenate(pieces, axis=0), full)


def test_shard_dataset_propagates_through_select(tabular_store):
    """Sharding reaches the leaf even under a select= wrapper.

    SelectBase reads from self.dataset (not self.forward), so the clone must
    shard that reference too; otherwise a selected dataset would read the full,
    unsharded leaf.
    """
    ds = TabularZarr(tabular_store)
    selected = ds._subset(select=["v0", "v2"])

    shards = shard_dataset(selected, COUNT)

    index = max(range(len(selected)), key=lambda i: selected[i].shape[0])
    full = selected[index]
    pieces = [s[index] for s in shards]

    # Column selection preserved, and rows are actually sharded (1/N each).
    assert full.shape[1] == 2
    assert all(p.shape[1] == 2 for p in pieces)
    assert sum(p.shape[0] for p in pieces) == full.shape[0]
    assert max(p.shape[0] for p in pieces) < full.shape[0]  # genuinely split
    np.testing.assert_array_equal(np.concatenate(pieces, axis=0), full)


def test_shard_dataset_combined_rejected():
    """Combined datasets reject sharding."""
    from anemoi.datasets.usage.forwards import Combined

    # Call the method directly; constructing a Combined needs compatible
    # datasets, which is irrelevant to the rejection contract.
    with pytest.raises(ValueError, match="combined datasets"):
        Combined.shard_view(object(), 0, COUNT)


def test_shard_dataset_bad_count(tabular_store):
    ds = TabularZarr(tabular_store)
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            shard_dataset(ds, bad)


def test_shard_dataset_rejects_non_tabular():
    from anemoi.datasets.usage.dataset import Dataset

    # A minimal non-tabular Dataset: it inherits the base shard_view, which
    # rejects sharding. Abstract methods are cleared (after class creation, as
    # ABCMeta recomputes them) so it can be instantiated.
    class NotTabular(Dataset):
        pass

    NotTabular.__abstractmethods__ = frozenset()

    with pytest.raises(ValueError, match="only supported for tabular"):
        shard_dataset(NotTabular(), 4)


# ----------------------------------------------------------------------
# Sizing introspection + reassembly
# ----------------------------------------------------------------------


def test_unsharded_sizes_and_shard_sizes(tabular_store):
    ds = TabularZarr(tabular_store)
    shards = shard_dataset(ds, COUNT)

    unsharded = ds.unsharded_sizes
    assert unsharded.shape == (len(ds),)

    # unsharded_sizes[n] == rows of the unsharded window n.
    sample = max(range(len(ds)), key=lambda i: ds[i].shape[0])
    assert unsharded[sample] == ds[sample].shape[0]

    assert ds.total_size == int(unsharded.sum())

    # An unsharded dataset is a single shard.
    assert ds.shard_sizes.shape == (len(ds), 1)
    np.testing.assert_array_equal(ds.shard_sizes[:, 0], unsharded)

    # The N-way table is known to the shards / container.
    shard_sizes = shards.shard_sizes
    assert shard_sizes.shape == (len(ds), COUNT)
    np.testing.assert_array_equal(shard_sizes.sum(axis=1), unsharded)

    # Every shard reports the same global tables.
    for s in shards:
        np.testing.assert_array_equal(s.unsharded_sizes, unsharded)
        np.testing.assert_array_equal(s.shard_sizes, shard_sizes)
        assert s.total_size == ds.total_size
    assert shards.total_size == ds.total_size

    # Each shard's actual row count per window matches shard_sizes[:, i].
    for i, s in enumerate(shards):
        np.testing.assert_array_equal(
            [s[n].shape[0] for n in range(len(s))],
            shard_sizes[:, i],
        )


def _reassemble(unsharded, shards, a, b):
    """Rebuild the unsharded ds[a:b] array by scattering each shard's rows."""
    full = np.asarray(unsharded[a:b])
    big = np.zeros_like(full)
    for s in shards:
        arr = s[a:b]
        data = np.asarray(arr)
        for src, dst in zip(arr.boundaries, arr.unsharded_boundaries):
            big[dst] = data[src]
    return big, full


def test_reassembly_multi_window(tabular_store):
    ds = TabularZarr(tabular_store)
    shards = shard_dataset(ds, COUNT)

    # A window range that contains data.
    sample = max(range(len(ds)), key=lambda i: ds[i].shape[0])
    a, b = max(0, sample - 5), min(len(ds), sample + 5)

    big, full = _reassemble(ds, shards.shards, a, b)
    np.testing.assert_array_equal(big, full)
    assert big.shape[0] == int(ds.unsharded_sizes[a:b].sum())


def test_reassembly_single_window(tabular_store):
    ds = TabularZarr(tabular_store)
    shards = shard_dataset(ds, COUNT)

    n = max(range(len(ds)), key=lambda i: ds[i].shape[0])
    full = ds[n]
    big = np.zeros_like(np.asarray(full))
    for s in shards:
        arr = s[n]
        (src,) = arr.boundaries
        (dst,) = arr.unsharded_boundaries
        big[dst] = np.asarray(arr)[src]
    np.testing.assert_array_equal(big, np.asarray(full))


def test_unsharded_boundaries_unsharded_equals_boundaries(tabular_store):
    """For an unsharded view, unsharded_boundaries == boundaries."""
    ds = TabularZarr(tabular_store)
    n = max(range(len(ds)), key=lambda i: ds[i].shape[0])
    arr = ds[n : n + 3]
    assert arr.unsharded_boundaries == arr.boundaries


# ----------------------------------------------------------------------
# Container + single shard
# ----------------------------------------------------------------------


def test_sharded_tabular_container(tabular_store):
    ds = TabularZarr(tabular_store)
    shards = shard_dataset(ds, COUNT)

    assert isinstance(shards, ShardedTabular)
    assert len(shards) == COUNT
    assert shards.num_shards == COUNT
    assert all(isinstance(s, TabularZarr) for s in shards)
    assert [shards[i] for i in range(COUNT)] == list(shards)

    # Sibling links: each shard knows its COUNT-1 siblings (objects, not self).
    for i, s in enumerate(shards):
        assert s.shard_index == i
        assert s.num_shards == COUNT
        assert len(s.other_shards) == COUNT - 1
        assert s not in s.other_shards


def test_single_shard(tabular_store):
    ds = TabularZarr(tabular_store)
    shards = shard_dataset(ds, COUNT)
    one = single_shard(ds, COUNT, 3)

    assert one.shard_index == 3
    assert one.num_shards == COUNT
    # No sibling objects, but the full N-way tables are still known.
    assert one.other_shards is None
    np.testing.assert_array_equal(one.shard_sizes, shards.shard_sizes)
    assert one.total_size == ds.total_size

    # Same rows as the corresponding element of the full sharded list.
    n = max(range(len(ds)), key=lambda i: ds[i].shape[0])
    np.testing.assert_array_equal(np.asarray(one[n]), np.asarray(shards[3][n]))


def test_single_shard_bad_index(tabular_store):
    ds = TabularZarr(tabular_store)
    for bad in (-1, COUNT, COUNT + 1):
        with pytest.raises(ValueError, match="shard index"):
            single_shard(ds, COUNT, bad)


# ----------------------------------------------------------------------
# open_dataset end-to-end
# ----------------------------------------------------------------------


def test_open_dataset_sharding(tabular_store):
    from anemoi.datasets.usage.gridded import open_dataset

    shards = open_dataset(tabular_store, sharding=COUNT)
    assert isinstance(shards, ShardedTabular)
    assert len(shards) == COUNT
    assert shards.total_size == TabularZarr(tabular_store).total_size


def test_open_dataset_single_shard(tabular_store):
    from anemoi.datasets.usage.gridded import open_dataset

    one = open_dataset(tabular_store, sharding=COUNT, shard=2)
    assert one.shard_index == 2
    assert one.num_shards == COUNT
    assert one.other_shards is None


def test_open_dataset_shard_requires_sharding(tabular_store):
    from anemoi.datasets.usage.gridded import open_dataset

    with pytest.raises(ValueError, match="`shard` requires `sharding`"):
        open_dataset(tabular_store, shard=1)
