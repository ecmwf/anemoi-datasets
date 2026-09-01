# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import zarr

LOG = logging.getLogger(__name__)

ZarrFileNotFoundError = FileNotFoundError
zarr_append_mode = "a"


class S3Store(zarr.storage.ObjectStore):
    """We use our class to manage per bucket credentials"""

    def __init__(self, url: str) -> None:
        from obstore.store import S3Store

        try:
            from anemoi.utils.remote.s3 import s3_options
        except ImportError:
            from anemoi.utils.remote.s3 import _s3_options as s3_options

        store = S3Store.from_url(url, **s3_options(url))
        super().__init__(store=store, read_only=True)


def HTTPStore(url: str) -> zarr.storage.FsspecStore:
    return zarr.storage.FsspecStore.from_url(url)


DebugStore = zarr.storage.LoggingStore


LocalStore = zarr.storage.LocalStore


def nested_store(path: str) -> str:
    """Return a store that writes chunk keys in a nested directory layout.

    The zarr3 format already nests chunk keys (``c/0/0``), and the layout is
    chosen per array via ``chunk_key_encoding`` rather than by the store, so
    the path is returned unchanged.

    Parameters
    ----------
    path : str
        Path to the store.

    Returns
    -------
    str
        The path, unchanged.
    """
    return path


def lru_store_cache(store: "str | zarr.abc.store.Store", max_size: int) -> "zarr.abc.store.Store":
    """Wrap a store in an in-memory LRU cache.

    This is the zarr3 counterpart of zarr2's ``LRUStoreCache``: a ``CacheStore``
    backed by a ``MemoryStore``, evicting least recently used entries once
    ``max_size`` bytes are cached. ``CacheStore`` still lives in
    ``zarr.experimental``, so the store is returned unwrapped on the zarr3
    releases that predate it.

    Parameters
    ----------
    store : str or zarr.abc.store.Store
        The store to wrap. A path is turned into a ``LocalStore``.
    max_size : int
        Maximum size of the cache, in bytes.

    Returns
    -------
    zarr.abc.store.Store
        The cached store.
    """
    try:
        from zarr.experimental.cache_store import CacheStore
    except ImportError:
        LOG.warning("Store caching is not available with zarr %s, ignoring cache=%s.", zarr.__version__, max_size)
        return store

    if isinstance(store, str):
        store = LocalStore(store)

    return CacheStore(store, cache_store=zarr.storage.MemoryStore(), max_size=max_size)


def blosc_compressor(cname: str = "zstd", clevel: int = 3, shuffle: int = 2) -> "zarr.abc.codec.Codec":
    """Return a Blosc compressor for zarr3.

    Parameters
    ----------
    cname : str
        The Blosc compressor name.
    clevel : int
        The compression level.
    shuffle : int
        The shuffle mode (0=none, 1=byte, 2=bit).

    Returns
    -------
    zarr.abc.codec.Codec
        The Blosc compressor codec.
    """
    from zarr.codecs import BloscCodec

    shuffle_names = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}
    return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle_names[shuffle])
