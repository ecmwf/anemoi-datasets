# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import zarr

zarr_version = int(zarr.__version__.split(".")[0])

zarr_private_files = (".zgroup", ".zattrs", ".zarray", "zarr.json")

if zarr_version < 3:
    from .zarr2 import DebugStore, HTTPStore, S3Store, ZarrFileNotFoundError, blosc_compressor, zarr_append_mode

    MemoryStore = zarr.storage.MemoryStore
else:
    from .zarr3 import DebugStore, HTTPStore, S3Store, ZarrFileNotFoundError, blosc_compressor, zarr_append_mode

    MemoryStore = zarr.storage.MemoryStore

__all__ = [
    "DebugStore",
    "HTTPStore",
    "MemoryStore",
    "S3Store",
    "ZarrFileNotFoundError",
    "blosc_compressor",
    "zarr_append_mode",
    "zarr_private_files",
    "zarr_version",
]
