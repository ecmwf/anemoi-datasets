# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import numpy as np
import zarr

from anemoi.datasets import ZARR3_DOES_NOT_SUPPORT_DATETIME64

from .synchronise import NoSynchroniser
from .synchronise import Synchroniser

LOG = logging.getLogger(__name__)


def add_zarr_dataset(
    *,
    name,
    dtype=None,
    fill_value=None,
    zarr_root,
    shape=None,
    array=None,
    overwrite=True,
    dimensions=None,
    **kwargs,
):
    assert dimensions is not None, "Please pass dimensions to add_zarr_dataset."
    assert isinstance(dimensions, (tuple, list))

    if dtype is None:
        assert array is not None, (name, shape, array, dtype, zarr_root)
        dtype = array.dtype

    if shape is None:
        assert array is not None, (name, shape, array, dtype, zarr_root)
        shape = array.shape

    if array is not None:
        if dtype == np.dtype("datetime64[s]") and ZARR3_DOES_NOT_SUPPORT_DATETIME64:
            dtype = "int64"
            array = array.astype(dtype)
        assert array.shape == shape, (array.shape, shape)
        a = zarr_root.create_array(
            name,
            shape=shape,
            dtype=dtype,
            overwrite=overwrite,
            **kwargs,
        )
        a[...] = array
        a.attrs["_ARRAY_DIMENSIONS"] = dimensions
        return a

    if "fill_value" not in kwargs:
        if str(dtype).startswith("float") or str(dtype).startswith("numpy.float"):
            kwargs["fill_value"] = np.nan
        elif str(dtype).startswith("datetime64") or str(dtype).startswith("numpy.datetime64"):
            kwargs["fill_value"] = np.datetime64("NaT")
        # elif str(dtype).startswith("timedelta64") or str(dtype).startswith(
        #    "numpy.timedelta64"
        # ):
        #    kwargs["fill_value"] = np.timedelta64("NaT")
        elif str(dtype).startswith("int") or str(dtype).startswith("numpy.int"):
            kwargs["fill_value"] = 0
        elif str(dtype).startswith("bool") or str(dtype).startswith("numpy.bool"):
            kwargs["fill_value"] = False
        else:
            raise ValueError(f"No fill_value for dtype={dtype}")

    if dtype == np.dtype("datetime64[s]") and ZARR3_DOES_NOT_SUPPORT_DATETIME64:
        dtype = "int64"

    a = zarr_root.create_array(
        name,
        shape=shape,
        dtype=dtype,
        overwrite=overwrite,
        **kwargs,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = dimensions
    return a


class ZarrBuiltRegistry:
    name_lengths = "lengths"
    name_flags = "flags"
    lengths = None
    flags = None
    z = None

    def __init__(self, path, synchronizer_path=None, use_threads=False):

        assert isinstance(path, str), path
        self.zarr_path = path
        self.synchronizer = Synchroniser(synchronizer_path) if synchronizer_path else NoSynchroniser()

    def clean(self):
        self.synchronizer.clean()

    def _open_write(self):
        return zarr.open(self.zarr_path, mode="r+")

    def _open_read(self):
        return zarr.open(self.zarr_path, mode="r")

    def new_dataset(self, *args, **kwargs):
        with self.synchronizer:
            z = self._open_write()
            zarr_root = z["_build"]
            add_zarr_dataset(*args, zarr_root=zarr_root, overwrite=True, dimensions=("tmp",), **kwargs)
            del z

    def add_to_history(self, action, **kwargs):
        new = dict(
            action=action,
            timestamp=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
        )
        new.update(kwargs)

        with self.synchronizer:
            z = self._open_write()
            history = z.attrs.get("history", [])
            history.append(new)
            z.attrs["history"] = history
            del z

    def get_lengths(self):
        with self.synchronizer:
            z = self._open_read()
            lenghts = list(z["_build"][self.name_lengths][:])
            del z
            return lenghts

    def get_flags(self, **kwargs):
        with self.synchronizer:
            z = self._open_read(**kwargs)
            flags = list(z["_build"][self.name_flags][:])
            del z
            return flags

    def get_flag(self, i):
        with self.synchronizer:
            z = self._open_read()
            flag = z["_build"][self.name_flags][i]
            del z
            return flag

    def set_flag(self, i, value=True):
        with self.synchronizer:
            z = self._open_write()
            z.attrs["latest_write_timestamp"] = (
                datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
            )
            z["_build"][self.name_flags][i] = value
            del z

    def ready(self):
        return all(self.get_flags())

    def create(self, lengths, overwrite=False):
        self.new_dataset(name=self.name_lengths, array=np.array(lengths, dtype="i4"))
        self.new_dataset(name=self.name_flags, array=np.array([False] * len(lengths), dtype=bool))
        self.add_to_history("initialised")

    def reset(self, lengths):
        return self.create(lengths, overwrite=True)

    def add_provenance(self, name):
        with self.synchronizer:
            z = self._open_write()

            if name in z.attrs:
                return

            from anemoi.utils.provenance import gather_provenance_info

            z.attrs[name] = gather_provenance_info()
            del z
