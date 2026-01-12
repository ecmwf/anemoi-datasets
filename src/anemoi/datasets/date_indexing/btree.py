# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property

import numpy as np
import zarr
from lru import LRU

from ..caching import ChunksCache
from . import DateIndexing
from . import date_indexing_registry
from .ranges import DateRange

LOG = logging.getLogger(__name__)


class Page:
    def __init__(
        self,
        /,
        page_id: int,
        is_node: bool = False,
        left: int = 0,
        right: int = 0,
        entries: list[dict] | None = None,
    ):
        assert page_id > 0, page_id
        self.page_id = page_id
        self.is_node = is_node
        self.left = left
        self.right = right
        self.entries = [] if entries is None else entries
        self._keys = None

    @property
    def keys(self) -> list[int]:
        if self._keys is None or len(self._keys) != len(self.entries):
            self._keys = np.array([entry.key for entry in self.entries], dtype=np.int64)
        return self._keys

    @property
    def count(self) -> int:
        return len(self.entries)


class Entry:
    def __init__(self, /, key: int):
        self.key = key


class NodeEntry(Entry):
    def __init__(self, /, key: int, child_page: int):
        super().__init__(key=key)
        self.child_page = child_page


class LeafEntry(Entry):
    def __init__(self, /, key: int, value: tuple[int, int]):
        super().__init__(key=key)
        self.value = value


class ZarrBTree:
    """B-tree implementation using Zarr arrays for storage.
    Key: datetime (stored as int64 seconds since epoch)
    Value: pair of integers (int64, int64)

    Uses a single Zarr array to store all pages efficiently.
    Each row represents a page with fixed structure.
    """

    def __init__(
        self,
        path: str,
        name: str = "time_index",
        page_size: int = 256,
        mode: str = "r",
        chunk_sizes: int = 64 * 1024 * 1024,  # 64 MB
        page_cache_size: int = 4096,
        compressor=None,
    ):
        """Initialize B-tree with Zarr backend.

        Args:
            path: Path to Zarr directory
            page_size: Maximum entries per page
            mode: 'r' for read-only, 'w' for overwrite, 'a' for read-write
        """

        self.page_size = page_size
        self.max_entries = max(4, page_size)  # 4 because a b-tree needs at least 2 entries per node after a split
        self.path = path

        if isinstance(path, str):
            # Open or create Zarr store
            self.store = zarr.open_group(self.path, mode=mode)
        else:
            self.store = path

        if name in self.store and mode == "w":
            # Overwrite existing
            del self.store[name]

        if name in self.store:
            self.pages = self.store[name]
            self.page_size = self.pages.attrs["page_size"]
            self.max_entries = self.pages.attrs["max_entries"]
            self.chunk_sizes = self.pages.attrs["chunk_sizes"]
            LOG.info(f"Opened existing B-tree with page size {self.page_size} and max entries {self.max_entries}")
        else:

            # Single array stores all pages:
            # Column layout: [is_node, count, left, right,
            #                 key0, val0_a, val0_b, key1, val1_a, val1_b, ..., keyN, valN_a, valN_b]
            # Keys are int64 (datetime as seconds since epoch)
            # Values are pairs of int64
            # Each page row has: 4 metadata cols + (max_entries * 3) data cols
            cols_per_page = 4 + (self.max_entries * 3)

            row_size = cols_per_page * 8
            chunk = round(chunk_sizes / row_size)
            LOG.info(f"Creating new B-tree with page size {self.page_size} and max entries {self.max_entries}")
            LOG.info(f"Zarr array chunk size: {chunk} rows x {cols_per_page} cols")

            # Initialize with root page
            self.store.create_dataset(
                name,
                shape=(chunk, cols_per_page),
                chunks=(chunk, cols_per_page),
                dtype="int64",
                fill_value=0,
                # compressor=compressor,
            )
            # Initialize root page (page_id=1, is_node=0/leaf, count=0)
            self.pages = self.store[name]
            self.pages[0] = [0, 0, 0, 0] + [0] * (self.max_entries * 3)

            # Metadata
            self.pages.attrs["_ARRAY_DIMENSIONS"] = ["pages", "payloads"]

            self.pages.attrs["root_page_id"] = 1
            self.pages.attrs["number_of_rows"] = 1
            self.pages.attrs["page_size"] = self.page_size
            self.pages.attrs["max_entries"] = self.max_entries
            self.pages.attrs["chunk_sizes"] = chunk_sizes

        self.pages = ChunksCache(self.pages)
        self._number_of_rows = None
        self._page_cache = LRU(page_cache_size)

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass

    def flush(self):
        """Flush cached changes to Zarr array"""
        if hasattr(self.pages, "flush"):
            self.pages.flush()

    def _page_id_to_row(self, page_id: int) -> int:
        """Convert page ID to row index"""
        return page_id - 1

    def _row_to_page_id(self, row: int) -> int:
        """Convert row index to page ID"""
        return row + 1

    def _read_page(self, page_id: int) -> dict:
        """Read a page from Zarr array"""

        if page_id in self._page_cache:
            return self._page_cache[page_id]

        row = self._page_id_to_row(page_id)
        if not (row >= 0 and row < self.pages.shape[0]):
            raise ValueError(f"Page {page_id} not found")

        data = self.pages[int(row), :]  # Cast to int for ChunksCache

        is_node = data[0]
        count = data[1]
        left = data[2]
        right = data[3]

        entries = []

        # Read entries (key and value pair)
        for i in range(count):
            key_idx = 4 + (i * 3)
            val_a_idx = 4 + (i * 3) + 1

            if is_node:
                # For nodes, val_a is child page_id, val_b is unused
                entries.append(NodeEntry(key=int(data[key_idx]), child_page=int(data[val_a_idx])))
            else:
                # For leaves, val_a and val_b are the actual value pair
                val_b_idx = 4 + (i * 3) + 2
                entries.append(
                    LeafEntry(
                        key=int(data[key_idx]),
                        value=(int(data[val_a_idx]), int(data[val_b_idx])),
                    )
                )

        page = Page(
            page_id=page_id,
            is_node=is_node,
            left=left,
            right=right,
            entries=entries,
        )

        self._page_cache[page_id] = page

        return page

    def _write_page(self, page: Page):
        """Write a page to Zarr array"""
        row = self._page_id_to_row(page.page_id)
        if not (row >= 0 and row < self.number_of_rows):
            raise ValueError(f"Page {page.page_id} not found")

        # Build row data
        data = [
            int(page.is_node),
            page.count,
            page.left,
            page.right,
        ]

        # Add entries
        for entry in page.entries:
            data.append(entry.key)
            if page.is_node:
                data.append(entry.child_page)
                data.append(0)  # Unused for nodes
            else:
                data.append(entry.value[0])
                data.append(entry.value[1])

        # Pad with zeros
        if len(data) < self.pages.shape[1]:
            data += [0] * (self.pages.shape[1] - len(data))

        self.pages[int(row), :] = data  # Cast to int for ChunksCache
        self._page_cache[page.page_id] = page

    def _create_page(self, is_node: bool = False) -> dict:
        """Create a new empty page"""
        # Allocate new row

        if self.number_of_rows >= self.pages.shape[0]:
            # Resize Zarr array to add more pages
            self.pages.resize(self.pages.shape[0] + self.pages.chunks[0], self.pages.shape[1])

        new_row = self.number_of_rows
        self.number_of_rows += 1

        page_id = self._row_to_page_id(new_row)

        page = Page(page_id=page_id, is_node=is_node)

        self._write_page(page)
        return page

    def _find_child(self, page: Page, key: int) -> int:
        """Find which child page to follow in a node using binary search"""
        if page.count == 0:
            return page.left

        # Extract keys for binary search
        keys = page.keys

        # Find insertion point using bisect
        idx = np.searchsorted(keys, key, "right")

        if idx == 0:
            return page.left
        else:
            return page.entries[idx - 1].child_page

    def _search(self, page_id: int, key: int) -> tuple[int, int] | None:
        """Search for a key, return value pair if found"""
        page = self._read_page(page_id)

        if page.is_node:
            child_id = self._find_child(page, key)
            return self._search(child_id, key)

        # Leaf page - binary search in entries
        keys = page.keys
        idx = np.searchsorted(keys, key, "left")

        if idx < len(keys) and keys[idx] == key:
            return page.entries[idx].value

        return None

    def _insert_in_leaf(self, page: Page, key: int, value: tuple[int, int]) -> bool:
        """Insert key-value into a leaf page. Returns True if key was updated."""
        # Binary search for insertion point
        keys = page.keys
        idx = np.searchsorted(keys, key, "left")

        if idx < len(keys) and keys[idx] == key:
            # Update existing
            page.entries[idx].value = value
            self._write_page(page)
            return True

        # Insert new entry at correct position
        page.entries.insert(idx, LeafEntry(key=key, value=value))
        self._write_page(page)

        return False

    def _split_leaf(self, page: Page) -> tuple[dict, int]:
        """Split a full leaf page, return (new_page, split_key)"""
        mid = page.count // 2

        new_page = self._create_page(is_node=False)
        new_page.entries = page.entries[mid:]

        # Update linked list
        new_page.right = page.right
        new_page.left = page.page_id
        page.right = new_page.page_id

        page.entries = page.entries[:mid]
        split_key = new_page.entries[0].key

        self._write_page(page)
        self._write_page(new_page)

        return new_page, split_key

    def _split_node(self, page: Page) -> tuple[dict, int]:
        """Split a full node page, return (new_page, split_key)"""
        mid = page.count // 2

        new_page = self._create_page(is_node=True)

        # Split key goes up to parent
        split_key = page.entries[mid].key

        # Right entries go to new page
        new_page.entries = page.entries[mid + 1 :]
        new_page.left = page.entries[mid].child_page

        # Keep left entries in original page
        page.entries = page.entries[:mid]

        self._write_page(page)
        self._write_page(new_page)

        return new_page, split_key

    def _insert_in_node(self, page: Page, key: int, child_page_id: int):
        """Insert key and child pointer into a node page"""
        # Binary search for insertion point
        idx = np.searchsorted(page.keys, key, "left")

        page.entries.insert(idx, NodeEntry(key=key, child_page=child_page_id))
        self._write_page(page)

    def _insert_recursive(self, page_id: int, key: int, value: tuple[int, int]) -> tuple[int, int] | None:
        """Insert key-value pair recursively.
        Returns (split_key, new_page_id) if split occurred, None otherwise.
        """
        page = self._read_page(page_id)

        if page.is_node:
            # Recurse to child
            child_id = self._find_child(page, key)
            result = self._insert_recursive(child_id, key, value)

            if result:
                split_key, new_child_id = result

                # Check if this node is already full before inserting
                if page.count >= self.max_entries:
                    # Split the node first
                    new_page, up_key = self._split_node(page)

                    # Determine which page gets the new entry
                    if split_key < up_key:
                        # Insert into left (original) page
                        page = self._read_page(page_id)
                        self._insert_in_node(page, split_key, new_child_id)
                    else:
                        # Insert into right (new) page
                        self._insert_in_node(new_page, split_key, new_child_id)

                    return (up_key, new_page.page_id)
                else:
                    # Space available, insert normally
                    self._insert_in_node(page, split_key, new_child_id)

            return None
        else:
            # Leaf page - check for existing key using binary search
            keys = page.keys
            idx = np.searchsorted(keys, key, "left")

            if idx < len(keys) and keys[idx] == key:
                # Update existing key (no split needed)
                page.entries[idx].value = value
                self._write_page(page)
                return None

            # Check if leaf is already full before inserting
            if page.count >= self.max_entries:
                # Split the leaf first
                new_page, split_key = self._split_leaf(page)

                # Determine which page gets the new entry
                if key < split_key:
                    # Insert into left (original) page
                    page = self._read_page(page_id)
                    self._insert_in_leaf(page, key, value)
                else:
                    # Insert into right (new) page
                    self._insert_in_leaf(new_page, key, value)

                return (split_key, new_page.page_id)
            else:
                # Space available, insert normally
                self._insert_in_leaf(page, key, value)
                return None

    @property
    def root_page_id(self) -> int:
        return self.pages.attrs["root_page_id"]

    @root_page_id.setter
    def root_page_id(self, value: int):
        self.pages.attrs["root_page_id"] = value

    @property
    def number_of_rows(self) -> int:
        if self._number_of_rows is None:
            self._number_of_rows = self.pages.attrs["number_of_rows"]
        return self._number_of_rows

    @number_of_rows.setter
    def number_of_rows(self, value: int):
        self.pages.attrs["number_of_rows"] = value
        self._number_of_rows = value

    def set(self, key_int: int, value: tuple[int, int]):
        """Set a key-value pair where value is a tuple of two integers"""
        root_id = self.root_page_id

        result = self._insert_recursive(root_id, key_int, value)

        if result:
            # Root needs to split - create new root
            split_key, new_page_id = result

            old_root_id = root_id
            new_root = self._create_page(is_node=True)

            new_root.left = old_root_id
            new_root.entries = [NodeEntry(key=split_key, child_page=new_page_id)]

            self._write_page(new_root)
            self.root_page_id = new_root.page_id

    def get(self, key: int) -> tuple[int, int] | None:
        """Get value pair for a key"""
        root_id = self.root_page_id
        return self._search(root_id, key)

    def _find_leaf_for_key(self, page_id: int, key: int) -> int:
        """Find the leaf page that would contain the given key"""
        page = self._read_page(page_id)

        if page.is_node:
            child_id = self._find_child(page, key)
            return self._find_leaf_for_key(child_id, key)
        else:
            return page_id

    def range(self, key1: int, key2: int) -> list[tuple[int, tuple[int, int]]]:
        """Get all entries in key range"""

        results = []
        self._range_search(self.root_page_id, key1, key2, results)
        return results

    def _range_search(
        self,
        page_id: int,
        key1: int,
        key2: int,
        results: list[tuple[int, tuple[int, int]]],
    ):
        """Collect all entries in range [key1, key2] using leaf linked list"""
        # Find the leaf page containing or after key1
        start_leaf_id = self._find_leaf_for_key(page_id, key1)

        # Traverse leaves using the linked list until we pass key2
        current_page = self._read_page(start_leaf_id)

        while True:
            # Process entries in current leaf
            for entry in current_page.entries:
                if entry.key > key2:
                    # Past the end of range, stop
                    return
                if entry.key >= key1:
                    # Within range, add to results
                    results.append((entry.key, entry.value))

            # Move to next leaf if it exists
            if current_page.right > 0:
                current_page = self._read_page(current_page.right)
            else:
                # No more leaves
                return

    def boundaries(self, key1, key2):
        """Get only the first and last entries in a key range.
        Much faster than range() when you only need the boundaries.

        Returns (first_entry, last_entry) where each entry is (datetime, (val_a, val_b))
        Returns (None, None) if range is empty.
        """

        root_id = self.root_page_id

        # Find first entry >= key1
        first_entry = self._find_first_in_range(root_id, key1, key2)
        if first_entry is None:
            return (None, None)

        # Find last entry <= key2
        last_entry = self._find_last_in_range(root_id, key1, key2)

        return (first_entry, last_entry)

    def _find_first_in_range(self, page_id: int, key1: int, key2: int):
        """Find the first entry in range [key1, key2]"""
        page = self._read_page(page_id)

        if page.is_node:
            # Navigate to the leaf that would contain key1
            child_id = self._find_child(page, key1)
            return self._find_first_in_range(child_id, key1, key2)

        # We're at a leaf - find first entry >= key1 and <= key2
        for entry in page.entries:
            if entry.key >= key1:
                if entry.key <= key2:
                    return (entry.key, entry.value)
                else:
                    # Past the range
                    return None

        # All entries in this leaf are < key1, try next leaf
        if page.right > 0:
            next_page = self._read_page(page.right)
            for entry in next_page.entries:
                if entry.key <= key2:
                    return (entry.key, entry.value)
                else:
                    return None

        return None

    def _find_last_in_range(self, page_id: int, key1: int, key2: int):
        """Find the last entry in range [key1, key2]"""
        page = self._read_page(page_id)

        if page.is_node:
            # Navigate to the leaf that would contain key2
            child_id = self._find_child(page, key2)
            return self._find_last_in_range(child_id, key1, key2)

        # We're at a leaf - find last entry <= key2 and >= key1
        # Search backwards for efficiency
        for i in range(page.count - 1, -1, -1):
            entry = page.entries[i]
            if entry.key <= key2:
                if entry.key >= key1:
                    return (entry.key, entry.value)
                else:
                    # Before the range
                    return None

        # All entries in this leaf are > key2, try previous leaf
        if page.left > 0:
            prev_page = self._read_page(page.left)
            for i in range(prev_page.count - 1, -1, -1):
                entry = prev_page.entries[i]
                if entry.key >= key1:
                    return (entry.key, entry.value)
                else:
                    return None

        return None

    def first_last_keys(self):
        """Get both the first (minimum) and last (maximum) keys in the tree.
        More efficient than calling first_key() and last_key() separately.
        Returns (first_key, last_key)
        """

        root = self._read_page(self.root_page_id)

        # Find first key by following leftmost path
        page = root
        while page.is_node:
            if page.left > 0:
                page = self._read_page(page.left)
            elif page.count > 0:
                page = self._read_page(page.entries[0].child_page)
            else:
                return (None, None)  # Empty tree

        if page.count == 0:
            return (None, None)  # Empty tree

        first_key = page.entries[0].key

        # Find last key by following rightmost path from root
        page = root
        while page.is_node:
            if page.count > 0:
                page = self._read_page(page.entries[-1].child_page)
            elif page.left > 0:
                page = self._read_page(page.left)
            else:
                # This shouldn't happen if we found a first_key, but handle it
                return (first_key, first_key)

        # Follow right links to find rightmost leaf
        while page.right > 0:
            page = self._read_page(page.right)

        assert page.count >= 0

        last_key = page.entries[-1].key

        return (first_key, last_key)

    def dump(self, page_id: int = None, depth: int = 0, label=lambda x: x):
        """Print tree structure"""
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)
        indent = "  " * depth

        page_type = "NODE" if page.is_node else "LEAF"
        print(f"{indent}{page_type} [id={page.page_id}, count={page.count} right={page.right}]")

        if not page.is_node:
            for entry in page.entries:
                print(f"{indent}  {label(entry.key)} -> {entry.value}")
        else:
            if page.left > 0:
                self.dump(page.left, depth + 1, label)
            for entry in page.entries:
                print(f"{indent}  key: {label(entry.key)}")
                self.dump(entry.child_page, depth + 1, label)

    def height(self, page_id: int = None, depth: int = 0):
        """Print tree structure"""
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        result = depth

        if not page.is_node:
            return result
        else:
            if page.left > 0:
                result = max(result, self.height(page.left, depth + 1))
            for entry in page.entries:
                result = max(result, self.height(entry.child_page, depth + 1))

        return result

    def size(self, page_id: int = None, depth: int = 0):
        """Print tree structure"""
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        result = 0

        if not page.is_node:
            return len(page.entries)
        else:
            if page.left > 0:
                result += self.size(page.left, depth + 1)
            for entry in page.entries:
                result += self.size(entry.child_page, depth + 1)

        return result

    def count(self, page_id: int = None, depth: int = 0):
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        result = 0

        if not page.is_node:
            return sum(entry.value[1] for entry in page.entries)
        else:
            if page.left > 0:
                result += self.count(page.left, depth + 1)
            for entry in page.entries:
                result += self.count(entry.child_page, depth + 1)

        return result

    def _iter(self, page_id: int = None, depth: int = 0):
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        if not page.is_node:
            yield from page.entries
        else:
            if page.left > 0:
                yield from self._iter(page.left, depth + 1)
            for entry in page.entries:
                yield from self._iter(entry.child_page, depth + 1)

    def __iter__(self):
        yield from self._iter()

    def bulk_load(self, data: np.ndarray) -> None:
        """Bulk load data into an empty B-tree.
        Much faster than inserting one-by-one for large datasets.
        Assumes that data is already sorted.

        Args:
            data: 2D numpy array of shape (n, 3) with dtype int64
                  Column 0: keys (datetime as microseconds since epoch)
                  Column 1: value first component
                  Column 2: value second component

        The data MUST be sorted by key (column 0) in ascending order.
        The tree must be empty before calling this method.
        """
        if data.shape[1] != 3:
            raise ValueError("Data must have exactly 3 columns: [key, val_a, val_b]")

        if len(data) == 0:
            return

        # Verify tree is empty (only root exists with no entries)
        root = self._read_page(self.root_page_id)
        if root.count > 0 or root.is_node:
            raise ValueError("Bulk load requires an empty tree. Current tree has data.")

        # Verify tree is empty (only root exists with no entries)
        root_id = self.root_page_id
        root = self._read_page(root_id)

        if root.count > 0 or root.is_node:
            raise ValueError("Bulk load requires an empty tree. Current tree has data.")

        # Clear page cache, as we will rewrite all pages
        self._page_cache.clear()

        # Verify data is sorted
        if not np.all(data[:-1, 0] <= data[1:, 0]):
            raise ValueError("Data must be sorted by key (column 0) in ascending order")

        n_entries = len(data)
        entries_per_leaf = self.max_entries
        n_leaves = (n_entries + entries_per_leaf - 1) // entries_per_leaf

        cols_per_page = self.pages.shape[1]

        # Special case: if all data fits in root leaf, write directly to existing root
        if n_leaves == 1:
            page_data = np.zeros(cols_per_page, dtype=np.int64)
            page_data[0] = 0  # is_node = False
            page_data[1] = n_entries  # count
            page_data[2] = 0  # left
            page_data[3] = 0  # right

            # Copy entries
            for j in range(n_entries):
                base_idx = 4 + (j * 3)
                page_data[base_idx] = int(data[j, 0])  # key
                page_data[base_idx + 1] = int(data[j, 1])  # val_a
                page_data[base_idx + 2] = int(data[j, 2])  # val_b

            # Write to the existing root page
            root_row = self._page_id_to_row(root_id)
            self.pages[root_row, :] = page_data

            # Verify the write worked
            verify = self.pages[root_row, :]
            assert verify[1] == n_entries, f"Write verification failed: count is {verify[1]}, expected {n_entries}"
            self.flush()
            return

        # Multi-leaf case: calculate total pages needed
        total_pages = 1 + n_leaves  # Keep existing root + new leaves starting after it
        current_level_size = n_leaves
        while current_level_size > 1:
            current_level_size = (current_level_size + self.max_entries) // (self.max_entries + 1)
            total_pages += current_level_size

        # Allocate all pages at once
        if total_pages > self.pages.shape[0]:
            self.pages.resize(total_pages, cols_per_page)

        # Build leaf pages starting after the existing root
        # If root is at page 1 (row 0), start at row 1
        leaf_page_ids = []
        leaf_first_keys = []
        current_row = self._page_id_to_row(root_id) + 1

        for i in range(0, n_entries, entries_per_leaf):
            end_idx = min(i + entries_per_leaf, n_entries)
            chunk = data[i:end_idx]
            chunk_size = len(chunk)

            page_id = self._row_to_page_id(current_row)

            # Build page data directly as numpy array
            page_data = np.zeros(cols_per_page, dtype=np.int64)
            page_data[0] = 0  # is_node = False
            page_data[1] = chunk_size  # count

            # Left link to previous leaf (if exists)
            if len(leaf_page_ids) > 0:
                page_data[2] = leaf_page_ids[-1]
            else:
                page_data[2] = 0

            # Right link to next leaf (if will exist)
            if end_idx < n_entries:
                page_data[3] = page_id + 1
            else:
                page_data[3] = 0

            # Copy entries directly: [key, val_a, val_b, key, val_a, val_b, ...]
            for j in range(chunk_size):
                base_idx = 4 + (j * 3)
                page_data[base_idx] = int(chunk[j, 0])  # key
                page_data[base_idx + 1] = int(chunk[j, 1])  # val_a
                page_data[base_idx + 2] = int(chunk[j, 2])  # val_b

            # Write entire page in one operation
            self.pages[current_row, :] = page_data

            leaf_page_ids.append(page_id)
            leaf_first_keys.append(int(chunk[0, 0]))
            current_row += 1

        # Build internal node levels bottom-up
        current_level_ids = leaf_page_ids
        current_level_keys = leaf_first_keys

        while len(current_level_ids) > 1:
            next_level_ids = []
            next_level_keys = []

            # Group current level into parent nodes
            nodes_per_page = self.max_entries

            for i in range(0, len(current_level_ids), nodes_per_page + 1):
                page_id = self._row_to_page_id(current_row)

                # Build node page data directly
                page_data = np.zeros(cols_per_page, dtype=np.int64)
                page_data[0] = 1  # is_node = True
                page_data[2] = current_level_ids[i]  # left child
                page_data[3] = 0  # right (unused for nodes)

                # Add entries: remaining children
                end_idx = min(i + nodes_per_page + 1, len(current_level_ids))
                entry_count = 0

                for j in range(i + 1, end_idx):
                    base_idx = 4 + (entry_count * 3)
                    page_data[base_idx] = current_level_keys[j]  # key
                    page_data[base_idx + 1] = current_level_ids[j]  # child_page
                    page_data[base_idx + 2] = 0  # unused
                    entry_count += 1

                page_data[1] = entry_count  # count

                # Write entire page in one operation
                self.pages[current_row, :] = page_data

                next_level_ids.append(page_id)
                next_level_keys.append(current_level_keys[i])
                current_row += 1

            current_level_ids = next_level_ids
            current_level_keys = next_level_keys

        # Update root to point to the top of the tree
        self.root_page_id = current_level_ids[0]

        self.flush()


@date_indexing_registry.register("btree")
class DateBTree(DateIndexing):
    name = "btree"

    def __init__(self, /, store, mode="r"):
        self.store = store
        self.mode = mode

    def bulk_load(self, dates_ranges: np.ndarray) -> None:
        btree = ZarrBTree(path=self.store, name="date_index_btree", mode=self.mode)
        btree.bulk_load(dates_ranges)

    @cached_property
    def start_end_epochs(self) -> tuple[int, int]:
        first_key, last_key = self.btree.first_last_keys()
        return first_key, last_key

    def start_end_dates(self) -> tuple[datetime.datetime, datetime.datetime]:
        first_key, last_key = self.start_end_epochs
        return datetime.datetime.fromtimestamp(first_key), datetime.datetime.fromtimestamp(last_key)

    def boundaries(self, start: int, end: int) -> tuple[int, int]:
        return self.btree.boundaries(start, end)

    @cached_property
    def btree(self) -> ZarrBTree:
        return ZarrBTree(path=self.store, name="date_index_btree", mode="r")

    def range_search(self, start: int, end: int, dataset_length: int) -> slice:
        start_entry, end_entry = self.btree.boundaries(start, end)
        if start_entry is None and end_entry is None:
            _, last_key = self.start_end_epochs
            # assert False, (start_entry, end_entry, start, end, last_key)
            if start > last_key:
                # Empty slice at end
                return slice(dataset_length, dataset_length)
            else:
                # Empty slice at start
                return slice(0, 0)

        start_entry = DateRange(start_entry[0], start_entry[1][0], start_entry[1][1])
        end_entry = DateRange(end_entry[0], end_entry[1][0], end_entry[1][1])

        print(f"Range search: start_entry={start_entry}, end_entry={end_entry}")

        return slice(start_entry.offset, end_entry.offset + end_entry.length)
