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


class Zarr2:
    @classmethod
    def base_store(cls):
        return zarr.storage.BaseStore

    @classmethod
    def is_zarr_group(cls, obj):
        return isinstance(obj, zarr.hierarchy.Group)

    @classmethod
    def create_array(cls, zarr_root, *args, **kwargs):
        return zarr_root.create_dataset(*args, **kwargs)

    @classmethod
    def change_dtype_datetime64(cls, dtype):
        return dtype

    @classmethod
    def cast_dtype_datetime64(cls, array, dtype):
        return array, dtype

    @classmethod
    def get_not_found_exception(cls):
        return zarr.errors.PathNotFoundError

    @classmethod
    def zarr_open_mode_append(cls):
        return "w+"

    @classmethod
    def zarr_open_to_patch_in_tests(cls):
        return "zarr.convenience.open"

    @classmethod
    def zarr_open(cls, *args, **kwargs):
        return zarr.convenience.open(*args, **kwargs)

    @classmethod
    def get_read_only_store_class(cls):
        class ReadOnlyStore(zarr.storage.BaseStore):
            """A base class for read-only stores."""

            def __delitem__(self, key: str) -> None:
                """Prevent deletion of items."""
                raise NotImplementedError()

            def __setitem__(self, key: str, value: bytes) -> None:
                """Prevent setting of items."""
                raise NotImplementedError()

            def __len__(self) -> int:
                """Return the number of items in the store."""
                raise NotImplementedError()

            def __iter__(self) -> iter:
                """Return an iterator over the store."""
                raise NotImplementedError()

        return ReadOnlyStore

    @classmethod
    def raise_if_not_supported(cls, msg):
        pass


class Zarr3:
    @classmethod
    def base_store(cls):
        return zarr.abc.store.Store

    @classmethod
    def is_zarr_group(cls, obj):
        return isinstance(obj, zarr.Group)

    @classmethod
    def create_array(cls, zarr_root, *args, **kwargs):
        if "compressor" in kwargs and kwargs["compressor"] is None:
            # compressor is deprecated, use compressors instead
            kwargs.pop("compressor")
            kwargs["compressors"] = ()
        return zarr_root.create_array(*args, **kwargs)

    @classmethod
    def get_not_found_exception(cls):
        return FileNotFoundError

    @classmethod
    def zarr_open_mode_append(cls):
        return "a"

    @classmethod
    def change_dtype_datetime64(cls, dtype):
        # remove this flag (and the relevant code) when Zarr 3 supports datetime64
        # https://github.com/zarr-developers/zarr-python/issues/2616
        import numpy as np

        if dtype == "datetime64[s]":
            dtype = np.dtype("int64")
        return dtype

    @classmethod
    def cast_dtype_datetime64(cls, array, dtype):
        # remove this flag (and the relevant code) when Zarr 3 supports datetime64
        # https://github.com/zarr-developers/zarr-python/issues/2616
        import numpy as np

        if dtype == np.dtype("datetime64[s]"):
            dtype = "int64"
            array = array.astype(dtype)

        return array, dtype

    @classmethod
    def zarr_open_to_patch_in_tests(cls):
        return "zarr.open"

    @classmethod
    def zarr_open(cls, *args, **kwargs):
        return zarr.open(*args, **kwargs)

    @classmethod
    def get_read_only_store_class(cls):
        class ReadOnlyStore(zarr.abc.store.Store):
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("Zarr 3 is not for this kind of store : {}".format(args))

        return ReadOnlyStore

    @classmethod
    def raise_if_not_supported(cls, msg="Zarr 3 is not supported in this context"):
        raise NotImplementedError(msg)


if zarr.__version__.startswith("3"):
    Zarr2AndZarr3 = Zarr3
else:
    LOG.warning("Using Zarr 2 : only zarr datasets build with zarr 2 are supported")
    Zarr2AndZarr3 = Zarr2
