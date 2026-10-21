# (C) Copyright 2025-2026 Anemoi contributors.
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
from zarr import Group

ZarrFileNotFoundError = zarr.errors.PathNotFoundError


class _ReadOnlyStore(zarr.storage.BaseStore):
    """A base class for read-only stores."""

    def __delitem__(self, key: str) -> None:
        raise TypeError("Deleting items is not allowed")

    def __setitem__(self, key: str, value: bytes) -> None:
        raise TypeError("Setting items is not allowed")

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

    def listdir(self, path: str | None = None) -> list[str]:
        """List immediate children of a directory.

        fsspec reads links from a server's directory page, which is how
        Zarr lists an HTTP store. Links that only jump around that page
        are skipped.
        """
        from urllib.parse import urlparse

        import fsspec

        directory = self.url
        if not directory.endswith("/"):
            directory = directory + "/"
        if path:
            directory = directory + path.strip("/") + "/"

        entries = fsspec.filesystem("http").ls(directory, detail=False)
        entries = [entry for entry in entries if entry.startswith(directory)]

        folder = urlparse(directory).path
        names = []
        for entry in entries:
            relative = urlparse(entry).path.removeprefix(folder).strip("/")
            name = relative.split("/")[0]
            if name and name not in names:
                names.append(name)
        return names


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

    def getitems(self, keys: list[str], **kwargs) -> dict[str, bytes]:
        """Fetch multiple chunks in parallel — overrides Zarr's default
        sequential getitems() for significantly higher throughput on S3.
        """
        from anemoi.utils.remote.s3 import get_objects_parallel

        urls = [os.path.join(self.url, k) for k in keys]
        results = get_objects_parallel(urls)
        out = {}
        for key, data in zip(keys, results):
            out[key] = data
        return out


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
    # Translate the zarr3 ``create_array`` API onto zarr2's ``create_dataset``
    # so call sites can be written once against the zarr3 signature.

    # zarr3 renamed ``compressor`` to ``compressors``.
    if "compressors" in kwargs:
        kwargs["compressor"] = kwargs.pop("compressors")

    # zarr3 forbids passing both ``data`` and ``shape``; drop the redundant
    # ``shape`` as it is implied by ``data``.
    if kwargs.get("data") is not None:
        kwargs.pop("shape", None)

    return group.create_dataset(name, **kwargs)


Group.create_array = _monkey_patch_create_array


def zarr_format_kwargs(zarr_format: int) -> dict:
    """Return the keyword arguments selecting the on-disk format of a new zarr store.

    zarr-python 2 names this option ``zarr_version``. Its experimental ``zarr_version=3``
    implements a draft of the specification that is incompatible with Zarr format 3,
    so only format 2 is supported.

    Parameters
    ----------
    zarr_format : int
        The Zarr format to write.

    Returns
    -------
    dict
        The keyword arguments to pass to ``zarr.open_group``.
    """
    if zarr_format != 2:
        raise ValueError(f"zarr_format={zarr_format} requires zarr-python>=3 (installed: {zarr.__version__})")
    return {"zarr_version": 2}


def get_zarr_format(node: Group | zarr.core.Array) -> int:
    """Return the Zarr format of a group or array, always 2 with zarr-python 2."""
    return 2


def blosc_compressor(cname: str = "zstd", clevel: int = 3, shuffle: int = 2, *, zarr_format: int) -> "zarr.Blosc":
    """Return a Blosc compressor for zarr2.

    Parameters
    ----------
    cname : str
        The Blosc compressor name.
    clevel : int
        The compression level.
    shuffle : int
        The shuffle mode (0=none, 1=byte, 2=bit).
    zarr_format : int
        The Zarr format of the array the compressor is used for.

    Returns
    -------
    zarr.Blosc
        The Blosc compressor.
    """
    zarr_format_kwargs(zarr_format)
    return zarr.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
