# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import zarr

if zarr.__version__.startswith("2."):
    from .zarr2 import DebugStore
    from .zarr2 import HTTPStore
    from .zarr2 import S3Store
    from .zarr2 import ZarrFileNotFoundError
else:
    from .zarr3 import DebugStore
    from .zarr3 import HTTPStore
    from .zarr3 import S3Store
    from .zarr3 import ZarrFileNotFoundError

__all__ = ["ZarrFileNotFoundError", "HTTPStore", "S3Store", "DebugStore"]
