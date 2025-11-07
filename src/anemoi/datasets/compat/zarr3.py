# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

import zarr

ZarrFileNotFoundError = FileNotFoundError


class S3Store(zarr.storage.ObjectStore):
    """We use our class to manage per bucket credentials"""

    def __init__(self, url: str) -> None:
        """Initialize the S3Store with a URL."""

        self.url = url

    def __getitem__(self, key: str) -> bytes:
        """Retrieve an item from the store."""
        from anemoi.utils.remote.s3 import get_object

        try:
            return get_object(os.path.join(self.url, key))
        except FileNotFoundError:
            raise KeyError(key)


DebugStore = zarr.storage.LoggingStore
