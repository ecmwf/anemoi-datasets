# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from . import source_registry
from .xarray import XarraySourceBase


@source_registry.register("xarray_kerchunk")
class XarrayKerchunkSource(XarraySourceBase):
    """An Xarray data source that uses the `kerchunk` engine."""

    emoji = "🧱"

    def __init__(self, context, json, *args, **kwargs: dict):
        super().__init__(context, *args, **kwargs)

        self.path_or_url = "reference://"

        self.options = {
            "engine": "zarr",
            "backend_kwargs": {
                "consolidated": False,
                "storage_options": {
                    "fo": json,
                    "remote_protocol": "s3",
                    "remote_options": {"anon": True},
                },
            },
        }
