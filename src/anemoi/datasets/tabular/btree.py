# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from threading import Lock
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import zarr

from .caching import FullCache

LOG = logging.getLogger(__name__)


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
        page_size: int = 128,
        mode: str = "r",
        chunk_sizes: int = 1024 * 1024,
    ):
        """Initialize B-tree with Zarr backend.

        Args:
            path: Path to Zarr directory
            page_size: Maximum entries per page
            mode: 'r' for read-only, 'w' for overwrite, 'a' for read-write
        """

        self.page_size = page_size
        self.max_entries = max(4, page_size // 2)  # Minimum 4 for proper B-tree
        self.path = path

        if isinstance(path, str):
            # Open or create Zarr store
            self.store = zarr.open_group(self.path, mode=mode)
        else:
            self.store = path

        # Single array stores all pages:
        # Column layout: [page_id, is_node, count, left, right,
        #                 key0, val0_a, val0_b, key1, val1_a, val1_b, ..., keyN, valN_a, valN_b]
        # Keys are int64 (datetime as seconds since epoch)
        # Values are pairs of int64
        # Each page row has: 5 metadata cols + (max_entries * 3) data cols

        cols_per_page = 5 + (self.max_entries * 3)

        LOG.info(
            f"Initializing B-tree at {self.path}, page_size={self.page_size}, max_entries={self.max_entries}, cols_per_page={cols_per_page}"
        )

        if name in self.store:
            self.pages = self.store[name]
        else:

            row_size = cols_per_page * 8
            chunk = round(chunk_sizes / row_size)
            LOG.info(f"Output chunk size: {(chunk, cols_per_page)}")

            # Initialize with root page
            self.store.create_dataset(
                name,
                shape=(1, cols_per_page),
                chunks=(chunk, cols_per_page),
                dtype="int64",
                fill_value=0,
                # compressor=None,
            )
            # Initialize root page (page_id=1, is_node=0/leaf, count=0)
            self.pages = self.store[name]
            self.pages[0] = [1, 0, 0, 0, 0] + [0] * (self.max_entries * 3)

            # Metadata
            self.pages.attrs["next_page_id"] = 2
            self.pages.attrs["root_page_id"] = 1
            self.pages.attrs["_ARRAY_DIMENSIONS"] = ["pages", "payloads"]

        self.pages = FullCache(self.pages)
        self.lock = Lock()

    def flush(self):
        """Flush cached changes to Zarr array"""
        with self.lock:
            self.pages.flush()

    def _get_page_row(self, page_id: int) -> Optional[int]:
        """Find row index for a given page_id"""
        # Linear search through page_ids (first column)
        page_ids = self.pages[:, 0]
        rows = np.where(page_ids == page_id)[0]
        return int(rows[0]) if len(rows) > 0 else None

    def _read_page(self, page_id: int) -> dict:
        """Read a page from Zarr array"""
        row = self._get_page_row(page_id)
        if row is None:
            raise ValueError(f"Page {page_id} not found")

        data = self.pages[row, :]

        page = {
            "page_id": int(data[0]),
            "is_node": bool(data[1]),
            "count": int(data[2]),
            "left": int(data[3]),
            "right": int(data[4]),
            "entries": [],
        }

        # Read entries (key and value pair)
        for i in range(page["count"]):
            key_idx = 5 + (i * 3)
            val_a_idx = 5 + (i * 3) + 1
            val_b_idx = 5 + (i * 3) + 2

            if page["is_node"]:
                # For nodes, val_a is child page_id, val_b is unused
                page["entries"].append({"key": int(data[key_idx]), "child_page": int(data[val_a_idx])})
            else:
                # For leaves, val_a and val_b are the actual value pair
                page["entries"].append(
                    {
                        "key": int(data[key_idx]),
                        "value": (int(data[val_a_idx]), int(data[val_b_idx])),
                    }
                )

        return page

    def _write_page(self, page: dict):
        """Write a page to Zarr array"""
        row = self._get_page_row(page["page_id"])

        if row is None:
            # Allocate new row
            current_shape = self.pages.shape
            self.pages.resize(current_shape[0] + 1, current_shape[1])
            row = current_shape[0]

        # Build row data
        data = [
            page["page_id"],
            int(page["is_node"]),
            page["count"],
            page["left"],
            page["right"],
        ]

        # Add entries
        for entry in page["entries"]:
            data.append(entry["key"])
            if page["is_node"]:
                data.append(entry["child_page"])
                data.append(0)  # Unused for nodes
            else:
                data.append(entry["value"][0])
                data.append(entry["value"][1])

        # Pad with zeros
        while len(data) < self.pages.shape[1]:
            data.append(0)

        self.pages[row, :] = data

    def _create_page(self, is_node: bool = False) -> dict:
        """Create a new empty page"""
        page_id = self.pages.attrs["next_page_id"]
        self.pages.attrs["next_page_id"] = page_id + 1

        page = {
            "page_id": page_id,
            "is_node": is_node,
            "count": 0,
            "left": 0,
            "right": 0,
            "entries": [],
        }

        self._write_page(page)
        return page

    def _find_child(self, page: dict, key: int) -> int:
        """Find which child page to follow in a node"""
        if page["count"] == 0:
            return page["left"]

        if key < page["entries"][0]["key"]:
            return page["left"]

        # Find the rightmost entry where entry.key <= key
        for i in range(page["count"] - 1, -1, -1):
            if key >= page["entries"][i]["key"]:
                return page["entries"][i]["child_page"]

        return page["left"]

    def _search(self, page_id: int, key: int) -> Optional[Tuple[int, int]]:
        """Search for a key, return value pair if found"""
        page = self._read_page(page_id)

        if page["is_node"]:
            child_id = self._find_child(page, key)
            return self._search(child_id, key)

        # Leaf page - search entries
        for entry in page["entries"]:
            if entry["key"] == key:
                return entry["value"]

        return None

    def _insert_in_leaf(self, page: dict, key: int, value: Tuple[int, int]) -> bool:
        """Insert key-value into a leaf page. Returns True if key was updated."""
        # Find insertion point
        insert_idx = page["count"]
        # keys = [entry["key"] for entry in page["entries"]]
        # insert_idx = bisect.bisect_left(keys, key)
        for i, entry in enumerate(page["entries"]):
            if entry["key"] == key:
                # Update existing
                entry["value"] = value
                self._write_page(page)
                return True
            if entry["key"] > key:
                insert_idx = i
                break

        # Insert new entry
        page["entries"].insert(insert_idx, {"key": key, "value": value})
        page["count"] += 1
        self._write_page(page)
        return False

    def _split_leaf(self, page: dict) -> Tuple[dict, int]:
        """Split a full leaf page, return (new_page, split_key)"""
        mid = page["count"] // 2

        new_page = self._create_page(is_node=False)
        new_page["entries"] = page["entries"][mid:]
        new_page["count"] = len(new_page["entries"])

        # Update linked list
        new_page["right"] = page["right"]
        new_page["left"] = page["page_id"]
        page["right"] = new_page["page_id"]

        page["entries"] = page["entries"][:mid]
        page["count"] = len(page["entries"])

        split_key = new_page["entries"][0]["key"]

        self._write_page(page)
        self._write_page(new_page)

        return new_page, split_key

    def _split_node(self, page: dict) -> Tuple[dict, int]:
        """Split a full node page, return (new_page, split_key)"""
        mid = page["count"] // 2

        new_page = self._create_page(is_node=True)

        # Split key goes up to parent
        split_key = page["entries"][mid]["key"]

        # Right entries go to new page
        new_page["entries"] = page["entries"][mid + 1 :]
        new_page["count"] = len(new_page["entries"])
        new_page["left"] = page["entries"][mid]["child_page"]

        # Keep left entries in original page
        page["entries"] = page["entries"][:mid]
        page["count"] = len(page["entries"])

        self._write_page(page)
        self._write_page(new_page)

        return new_page, split_key

    def _insert_in_node(self, page: dict, key: int, child_page_id: int):
        """Insert key and child pointer into a node page"""
        insert_idx = page["count"]
        for i, entry in enumerate(page["entries"]):
            if entry["key"] > key:
                insert_idx = i
                break

        page["entries"].insert(insert_idx, {"key": key, "child_page": child_page_id})
        page["count"] += 1
        self._write_page(page)

    def _insert_recursive(self, page_id: int, key: int, value: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Insert key-value pair recursively.
        Returns (split_key, new_page_id) if split occurred, None otherwise.
        """
        page = self._read_page(page_id)

        if page["is_node"]:
            # Recurse to child
            child_id = self._find_child(page, key)
            result = self._insert_recursive(child_id, key, value)

            if result:
                split_key, new_child_id = result

                # Insert split key into this node
                self._insert_in_node(page, split_key, new_child_id)
                page = self._read_page(page_id)  # Reload after modification

                # Check if this node needs splitting
                if page["count"] >= self.max_entries:
                    new_page, up_key = self._split_node(page)
                    return (up_key, new_page["page_id"])

            return None
        else:
            # Leaf page
            updated = self._insert_in_leaf(page, key, value)

            if updated:
                return None  # Just updated existing key, no split

            page = self._read_page(page_id)  # Reload

            # Check if split needed
            if page["count"] >= self.max_entries:
                new_page, split_key = self._split_leaf(page)
                return (split_key, new_page["page_id"])

            return None

    @property
    def root_page_id(self) -> int:
        return self.pages.attrs["root_page_id"]

    @root_page_id.setter
    def root_page_id(self, value: int):
        self.pages.attrs["root_page_id"] = value

    def set(self, key_int: int, value: Tuple[int, int]):
        """Set a key-value pair where value is a tuple of two integers"""
        root_id = self.root_page_id

        result = self._insert_recursive(root_id, key_int, value)

        if result:
            # Root needs to split - create new root
            split_key, new_page_id = result

            old_root_id = root_id
            new_root = self._create_page(is_node=True)

            new_root["count"] = 1
            new_root["left"] = old_root_id
            new_root["entries"] = [{"key": split_key, "child_page": new_page_id}]

            self._write_page(new_root)
            self.root_page_id = new_root["page_id"]

    def get(self, key: int) -> Optional[Tuple[int, int]]:
        """Get value pair for a key"""
        root_id = self.root_page_id
        return self._search(root_id, key)

    def _find_leaf_for_key(self, page_id: int, key: int) -> int:
        """Find the leaf page that would contain the given key"""
        page = self._read_page(page_id)

        if page["is_node"]:
            child_id = self._find_child(page, key)
            return self._find_leaf_for_key(child_id, key)
        else:
            return page_id

    def _range_search(
        self,
        page_id: int,
        key1: int,
        key2: int,
        results: List[Tuple[int, Tuple[int, int]]],
    ):
        """Collect all entries in range [key1, key2] using leaf linked list"""
        # Find the leaf page containing or after key1
        start_leaf_id = self._find_leaf_for_key(page_id, key1)

        # Traverse leaves using the linked list until we pass key2
        current_page = self._read_page(start_leaf_id)

        while True:
            # Process entries in current leaf
            for entry in current_page["entries"]:
                if entry["key"] > key2:
                    # Past the end of range, stop
                    return
                if entry["key"] >= key1:
                    # Within range, add to results
                    results.append((entry["key"], entry["value"]))

            # Move to next leaf if it exists
            if current_page["right"] > 0:
                current_page = self._read_page(current_page["right"])
            else:
                # No more leaves
                return

    def boundaries(self, key1: int, key2: int) -> Tuple[Tuple[int, Tuple[int, int]], Tuple[int, Tuple[int, int]]]:
        """Get all first and last entries in key range"""

        assert key1 <= key2, (key1, key2)

        class FirstLast:
            def __init__(self):
                self.first = None
                self.last = None

            def append(self, key):
                if self.first is None:
                    self.first = key
                self.last = key

        collect = FirstLast()
        self._range_search(self.root_page_id, key1, key2, collect)
        return collect.first, collect.last

    def range(self, key1: int, key2: int) -> List[Tuple[int, Tuple[int, int]]]:
        """Get all entries in key range"""

        results = []
        self._range_search(self.root_page_id, key1, key2, results)
        return results

    def start_end(self, page_id: int = None, depth: int = 0):
        """Print tree structure"""
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        start, end = None, None

        if not page["is_node"]:
            return (
                (page["entries"][0]["key"], page["entries"][0]["value"]),
                (page["entries"][-1]["key"], page["entries"][-1]["value"]),
            )
        else:
            if page["left"] > 0:
                start, end = self.start_end(page["left"], depth + 1)
            for entry in page["entries"]:
                s, e = self.start_end(entry["child_page"], depth + 1)
                start = min(start, s) if start else s
                end = max(end, e) if end else e

        return start, end

    def dump(self, page_id: int = None, depth: int = 0, label=lambda x: x):
        """Print tree structure"""
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)
        indent = "  " * depth

        page_type = "NODE" if page["is_node"] else "LEAF"
        print(f"{indent}{page_type} [id={page['page_id']}, count={page['count']} right={page['right']}]")

        if not page["is_node"]:
            for entry in page["entries"]:
                print(f"{indent}  {label(entry['key'])} -> {entry['value']}")
        else:
            if page["left"] > 0:
                self.dump(page["left"], depth + 1, label)
            for entry in page["entries"]:
                print(f"{indent}  key: {label(entry['key'])}")
                self.dump(entry["child_page"], depth + 1, label)

    def height(self, page_id: int = None, depth: int = 0):
        """Print tree structure"""
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        result = depth

        if not page["is_node"]:
            return result
        else:
            if page["left"] > 0:
                result = max(result, self.height(page["left"], depth + 1))
            for entry in page["entries"]:
                result = max(result, self.height(entry["child_page"], depth + 1))

        return result

    def size(self, page_id: int = None, depth: int = 0):
        """Print tree structure"""
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        result = 0

        if not page["is_node"]:
            return len(page["entries"])
        else:
            if page["left"] > 0:
                result += self.size(page["left"], depth + 1)
            for entry in page["entries"]:
                result += self.size(entry["child_page"], depth + 1)

        return result

    def count(self, page_id: int = None, depth: int = 0):
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        result = 0

        if not page["is_node"]:
            return sum(entry["value"][1] for entry in page["entries"])
        else:
            if page["left"] > 0:
                result += self.count(page["left"], depth + 1)
            for entry in page["entries"]:
                result += self.count(entry["child_page"], depth + 1)

        return result

    def _iter(self, page_id: int = None, depth: int = 0):
        if page_id is None:
            page_id = self.root_page_id

        page = self._read_page(page_id)

        if not page["is_node"]:
            yield from page["entries"]
        else:
            if page["left"] > 0:
                yield from self._iter(page["left"], depth + 1)
            for entry in page["entries"]:
                yield from self._iter(entry["child_page"], depth + 1)

    def __iter__(self):
        yield from self._iter()
