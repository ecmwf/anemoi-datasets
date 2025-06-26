# (C) Copyright 2025 Anemoi contributors.
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

version = 3
FileNotFoundException = FileNotFoundError
Group = zarr.Group
open_mode_append = "a"


class S3Store(zarr.storage.ObjectStore):
    """We use our class to manage per bucket credentials"""

    def __init__(self, url):

        import boto3
        from anemoi.utils.remote.s3 import s3_options
        from obstore.auth.boto3 import Boto3CredentialProvider
        from obstore.store import from_url

        options = s3_options(url)

        credential_provider = Boto3CredentialProvider(
            session=boto3.session.Session(
                aws_access_key_id=options["aws_access_key_id"],
                aws_secret_access_key=options["aws_secret_access_key"],
            ),
        )

        objectstore = from_url(
            url,
            credential_provider=credential_provider,
            endpoint=options["endpoint_url"],
        )

        super().__init__(objectstore, read_only=True)


class HTTPStore(zarr.storage.ObjectStore):

    def __init__(self, url):

        from obstore.store import from_url

        objectstore = from_url(url)

        super().__init__(objectstore, read_only=True)


DebugStore = zarr.storage.LoggingStore


class PlanetaryComputerStore:
    """We write our own Store to access catalogs on Planetary Computer,
    as it requires some extra arguments to use xr.open_zarr.
    """

    def __init__(self, data_catalog_id: str) -> None:
        """Initialize the PlanetaryComputerStore with a data catalog ID.

        Parameters
        ----------
        data_catalog_id : str
            The data catalog ID.
        """
        super().__init__()
        self.data_catalog_id = data_catalog_id

        import planetary_computer
        import pystac_client

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1/",
            modifier=planetary_computer.sign_inplace,
        )
        collection = catalog.get_collection(self.data_catalog_id)

        asset = collection.assets["zarr-abfs"]

        if "xarray:storage_options" in asset.extra_fields:
            store = {
                "store": asset.href,
                "storage_options": asset.extra_fields["xarray:storage_options"],
                **asset.extra_fields["xarray:open_kwargs"],
            }
        else:
            store = {
                "filename_or_obj": asset.href,
                **asset.extra_fields["xarray:open_kwargs"],
            }

        self.store = store


def create_array(zarr_root, *args, **kwargs):
    if "compressor" in kwargs and kwargs["compressor"] is None:
        # compressor is deprecated, use compressors instead
        kwargs.pop("compressor")
        kwargs["compressors"] = ()

    data = kwargs.pop("data", None)
    if data is not None:
        kwargs.setdefault("dtype", change_dtype_datetime64(data.dtype))
        kwargs.setdefault("shape", data.shape)

    try:
        z = zarr_root.create_array(*args, **kwargs)
        if data is not None:
            z[:] = data
        return z
    except Exception:
        LOG.exception("Failed to create array in Zarr store")
        LOG.error(
            "Failed to create array in Zarr store with args: %s, kwargs: %s",
            args,
            kwargs,
        )
        raise


def change_dtype_datetime64(dtype):
    # remove this flag (and the relevant code) when Zarr 3 supports datetime64
    # https://github.com/zarr-developers/zarr-python/issues/2616
    import numpy as np

    if dtype == "datetime64[s]":
        dtype = np.dtype("int64")
    return dtype


def cast_dtype_datetime64(array, dtype):
    # remove this flag (and the relevant code) when Zarr 3 supports datetime64
    # https://github.com/zarr-developers/zarr-python/issues/2616
    import numpy as np

    if dtype == np.dtype("datetime64[s]"):
        dtype = "int64"
        array = array.astype(dtype)

    return array, dtype


def supports_datetime64():
    store = zarr.storage.MemoryStore()
    try:
        zarr.create_array(store=store, shape=(10,), dtype="datetime64[s]")
        return True
    except KeyError:
        # If a KeyError is raised, it means datetime64 is not supported
        return False


if __name__ == "__main__":
    print("Zarr version:", version)
    print("Zarr supports datetime64:", supports_datetime64())
