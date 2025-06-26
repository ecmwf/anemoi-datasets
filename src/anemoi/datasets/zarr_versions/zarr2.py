# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging
import warnings
from typing import Any
from typing import Optional

import zarr

LOG = logging.getLogger(__name__)


version = 2

FileNotFoundException = zarr.errors.PathNotFoundError
Group = zarr.hierarchy.Group
open_mode_append = "w+"


class ReadOnlyStore(zarr.storage.BaseStore):
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


class S3Store(ReadOnlyStore):
    """A read-only store for S3 resources."""

    """We write our own S3Store because the one used by zarr (s3fs)
    does not play well with fork(). We also get to control the s3 client
    options using the anemoi configs.
    """

    def __init__(self, url: str, region: Optional[str] = None) -> None:
        """Initialize the S3Store with a URL and optional region."""
        from anemoi.utils.remote.s3 import s3_client

        super().__init__()

        _, _, self.bucket, self.key = url.split("/", 3)
        self.s3 = s3_client(self.bucket, region=region)

    # Version 2
    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item from the store."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key + "/" + key)
        except self.s3.exceptions.NoSuchKey:
            raise KeyError(key)

        return response["Body"].read()


class HTTPStore(ReadOnlyStore):
    """A read-only store for HTTP(S) resources."""

    def __init__(self, url: str) -> None:
        """Initialize the HTTPStore with a URL."""
        super().__init__()
        self.url = url

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item from the store."""
        import requests

        r = requests.get(self.url + "/" + key)

        if r.status_code == 404:
            raise KeyError(key)

        r.raise_for_status()
        return r.content


class PlanetaryComputerStore(ReadOnlyStore):
    """We write our own Store to access catalogs on Planetary Computer,
    as it requires some extra arguments to use xr.open_zarr.
    """

    def __init__(self, data_catalog_id: str) -> None:
        """Initialize the PlanetaryComputerStore with a data catalog ID.

        Parameters
        ----------
        data_catalog_id : str
            The data catalog ID.
        """
        super().__init__()
        self.data_catalog_id = data_catalog_id

        import planetary_computer
        import pystac_client

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1/",
            modifier=planetary_computer.sign_inplace,
        )
        collection = catalog.get_collection(self.data_catalog_id)

        asset = collection.assets["zarr-abfs"]

        if "xarray:storage_options" in asset.extra_fields:
            store = {
                "store": asset.href,
                "storage_options": asset.extra_fields["xarray:storage_options"],
                **asset.extra_fields["xarray:open_kwargs"],
            }
        else:
            store = {
                "filename_or_obj": asset.href,
                **asset.extra_fields["xarray:open_kwargs"],
            }

        self.store = store

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item from the store."""
        raise NotImplementedError()


class DebugStore(ReadOnlyStore):
    """A store to debug the zarr loading."""

    def __init__(self, store: Any) -> None:
        super().__init__()
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


def create_array(zarr_root, *args, **kwargs):
    return zarr_root.create_dataset(*args, **kwargs)


def change_dtype_datetime64(dtype):
    return dtype


def cast_dtype_datetime64(array, dtype):
    return array, dtype


def supports_datetime64():
    return True
