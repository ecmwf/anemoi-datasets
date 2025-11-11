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


@source_registry.register("planetary_computer")
class PlanetaryComputerSource(XarraySourceBase):
    """An Xarray data source for the planetary_computer."""

    emoji = "🪐"

    def __init__(self, context, data_catalog_id, version="v1", *args, **kwargs: dict):

        import planetary_computer
        import pystac_client

        self.data_catalog_id = data_catalog_id
        self.flavour = kwargs.pop("flavour", None)
        self.patch = kwargs.pop("patch", None)
        self.options = kwargs.pop("options", {})

        catalog = pystac_client.Client.open(
            f"https://planetarycomputer.microsoft.com/api/stac/{version}/",
            modifier=planetary_computer.sign_inplace,
        )
        collection = catalog.get_collection(self.data_catalog_id)

        asset = collection.assets["zarr-abfs"]

        if "xarray:storage_options" in asset.extra_fields:
            self.options["storage_options"] = asset.extra_fields["xarray:storage_options"]

        self.options.update(asset.extra_fields["xarray:open_kwargs"])

        super().__init__(context, url=asset.href, *args, **kwargs)
