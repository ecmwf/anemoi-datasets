# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import warnings

import zarr
from zarr.hierarchy import Group

ZarrFileNotFoundError = zarr.errors.PathNotFoundError
zarr_append_mode = "w+"


class _ReadOnlyStore(zarr.storage.BaseStore):
    """A base class for read-only stores."""

    def __delitem__(self, key: str) -> None:
        """Prevent deletion of items."""
        raise NotImplementedError()

    def __setitem__(self, key: str, value: bytes) -> None:
        """Prevent setting of items."""
        raise NotImplementedError()

    def __len__(self) -> int:
        """Return the number of items in the store."""
        raise NotImplementedError()

    def __iter__(self) -> iter:
        """Return an iterator over the store."""
        raise NotImplementedError()


class HTTPStore(_ReadOnlyStore):
    """A read-only store for HTTP(S) resources."""

    def __init__(self, url: str) -> None:
        """Initialize the HTTPStore with a URL."""
        self.url = url

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item from the store."""
        import requests

        r = requests.get(self.url + "/" + key, timeout=10)

        if r.status_code == 404:
            raise KeyError(key)

        r.raise_for_status()
        return r.content


class S3Store(_ReadOnlyStore):
    """A read-only store for S3 resources."""

    """We write our own S3Store because the one used by zarr (s3fs)
    does not play well with fork(). We also get to control the s3 client
    options using the anemoi configs.
    """

    def __init__(self, url: str) -> None:
        """Initialize the S3Store with a URL."""

        self.url = url

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item from the store."""
        from anemoi.utils.remote.s3 import get_object

        try:
            return get_object(os.path.join(self.url, key))
        except FileNotFoundError:
            raise KeyError(key)


class DebugStore(_ReadOnlyStore):
    """A store to debug the zarr loading."""

    def __init__(self, store: _ReadOnlyStore) -> None:
        """Initialize the DebugStore with another store."""
        assert not isinstance(store, DebugStore)
        self.store = store

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item from the store and print debug information."""
        # print()
        print("GET", key, self)
        # traceback.print_stack(file=sys.stdout)
        return self.store[key]

    def __len__(self) -> int:
        """Return the number of items in the store."""
        return len(self.store)

    def __iter__(self) -> iter:
        """Return an iterator over the store."""
        warnings.warn("DebugStore: iterating over the store")
        return iter(self.store)

    def __contains__(self, key: str) -> bool:
        """Check if the store contains a key."""
        return key in self.store


def _monkey_patch_create_array(
    group: Group,
    name: str,
    **kwargs,
) -> zarr.core.Array:
    return group.create_dataset(name, **kwargs)


Group.create_array = _monkey_patch_create_array
