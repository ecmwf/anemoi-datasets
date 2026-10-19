# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import zarr

# Major version of the installed zarr-python library. Not to be confused with zarr-python 2's
# ``zarr_version`` keyword argument, which selects the on-disk Zarr format.
zarr_version = int(zarr.__version__.split(".")[0])

zarr_private_files = (".zgroup", ".zattrs", ".zarray", "zarr.json")

# On-disk Zarr format of new stores. Keep at 2 until the readers (anemoi-training,
# anemoi-inference) support Zarr format 3.
DEFAULT_ZARR_FORMAT = 2

# Names of the Zarr formats in the user-facing API (recipe ``output.format``, ``copy --format``)
_FORMATS = {"zarr2": 2, "zarr3": 3}

DEFAULT_FORMAT = f"zarr{DEFAULT_ZARR_FORMAT}"


def zarr_format_from_name(name: str) -> int:
    """Return the Zarr format number for a user-facing format name.

    Parameters
    ----------
    name : str
        The format name, ``"zarr2"`` or ``"zarr3"``.

    Returns
    -------
    int
        The Zarr format, 2 or 3.
    """
    try:
        return _FORMATS[name]
    except KeyError:
        raise ValueError(f"Invalid format {name!r}, expected one of {', '.join(map(repr, _FORMATS))}") from None


if zarr_version < 3:
    from .zarr2 import DebugStore
    from .zarr2 import HTTPStore
    from .zarr2 import S3Store
    from .zarr2 import ZarrFileNotFoundError
    from .zarr2 import blosc_compressor
    from .zarr2 import get_zarr_format
    from .zarr2 import zarr_format_kwargs

    MemoryStore = zarr.storage.MemoryStore
else:
    from .zarr3 import DebugStore
    from .zarr3 import HTTPStore
    from .zarr3 import S3Store
    from .zarr3 import ZarrFileNotFoundError
    from .zarr3 import blosc_compressor
    from .zarr3 import get_zarr_format
    from .zarr3 import zarr_format_kwargs

    MemoryStore = zarr.storage.MemoryStore


def zarr_open(store: Any, mode: str, *, zarr_format: int | None = None) -> Any:
    """Open or create a zarr store. Use this instead of ``zarr.open`` or ``zarr.open_group``.

    New stores are written in ``zarr_format`` (default ``DEFAULT_ZARR_FORMAT``), never in the
    default format of the installed zarr-python (format 3 with zarr-python 3). The format of an
    existing store is detected.

    Parameters
    ----------
    store : Any
        A path or a zarr store.
    mode : str
        ``"r"`` or ``"r+"`` to open an existing store, read-only or read-write.
        ``"w"`` to create a group, replacing any existing store; ``"w-"`` to create a group, failing if one exists.
        ``"a"`` to open as ``"r+"`` if the store exists, otherwise create it as ``"w-"``.
    zarr_format : int, optional
        The Zarr format of a new store. When opening an existing store, its format is checked against it.

    Returns
    -------
    zarr.Group | zarr.Array
        The opened group or array. New stores are always groups.
    """
    if mode == "a":
        try:
            return zarr_open(store, "r+", zarr_format=zarr_format)
        except ZarrFileNotFoundError:
            # Probe with "r+" rather than passing "a" to zarr: with zarr-python 3, "a" and an
            # explicit format on an existing store of the other format creates a second root.
            mode = "w-"

    if mode in ("w", "w-"):
        # zarr.open would create an array in "w-" mode with zarr-python 3
        fmt = DEFAULT_ZARR_FORMAT if zarr_format is None else zarr_format
        return zarr.open_group(store, mode=mode, **zarr_format_kwargs(fmt))

    if mode in ("r", "r+"):
        node = zarr.open(store, mode=mode)
        if zarr_format is not None and get_zarr_format(node) != zarr_format:
            raise ValueError(f"{store} is in Zarr format {get_zarr_format(node)}, expected format {zarr_format}")
        return node

    raise ValueError(f"Unsupported mode {mode!r}, expected one of 'r', 'r+', 'w', 'w-' or 'a'")


__all__ = [
    "DEFAULT_FORMAT",
    "DEFAULT_ZARR_FORMAT",
    "ZarrFileNotFoundError",
    "HTTPStore",
    "S3Store",
    "DebugStore",
    "MemoryStore",
    "blosc_compressor",
    "get_zarr_format",
    "zarr_format_from_name",
    "zarr_open",
    "zarr_version",
    "zarr_private_files",
]
