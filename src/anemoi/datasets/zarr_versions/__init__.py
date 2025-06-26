# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import zarr

version = zarr.__version__.split(".")[0]

if version == "2":
    from . import zarr2 as zarr_2_or_3

elif version == "3":
    from . import zarr3 as zarr_2_or_3
else:
    raise ImportError(f"Unsupported Zarr version: {zarr.__version__}. Supported versions are 2 and 3.")

__all__ = ["zarr_2_or_3"]
