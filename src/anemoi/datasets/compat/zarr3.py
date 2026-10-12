# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numcodecs
import zarr

ZarrFileNotFoundError = FileNotFoundError


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


def zarr_format_kwargs(zarr_format: int) -> dict:
    """Return the keyword arguments selecting the on-disk format of a new zarr store.

    Parameters
    ----------
    zarr_format : int
        The Zarr format to write (2 or 3).

    Returns
    -------
    dict
        The keyword arguments to pass to ``zarr.open_group``.
    """
    if zarr_format not in (2, 3):
        raise ValueError(f"Unsupported zarr_format={zarr_format}, expected 2 or 3")
    return {"zarr_format": zarr_format}


def get_zarr_format(node: zarr.Group | zarr.Array) -> int:
    """Return the Zarr format of a group or array."""
    return node.metadata.zarr_format


def blosc_compressor(
    cname: str = "zstd", clevel: int = 3, shuffle: int = 2, *, zarr_format: int
) -> "zarr.abc.codec.Codec | numcodecs.abc.Codec":
    """Return a Blosc compressor for zarr3.

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
    zarr.abc.codec.Codec | numcodecs.abc.Codec
        The Blosc compressor codec. Zarr format 2 arrays only accept numcodecs codecs.
    """
    zarr_format_kwargs(zarr_format)

    if zarr_format == 2:
        return numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)

    from zarr.codecs import BloscCodec

    shuffle_names = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}
    return BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle_names[shuffle])
