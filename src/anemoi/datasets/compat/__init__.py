# (C) Copyright 2025 Anemoi contributors.
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
    from .zarr2 import DebugStore
    from .zarr2 import HTTPStore
    from .zarr2 import S3Store
    from .zarr2 import ZarrFileNotFoundError
    from .zarr2 import zarr_append_mode
else:
    from .zarr3 import DebugStore
    from .zarr3 import HTTPStore
    from .zarr3 import S3Store
    from .zarr3 import ZarrFileNotFoundError
    from .zarr3 import zarr_append_mode

__all__ = [
    "ZarrFileNotFoundError",
    "HTTPStore",
    "S3Store",
    "DebugStore",
    "zarr_append_mode",
    "zarr_version",
    "zarr_private_files",
]
