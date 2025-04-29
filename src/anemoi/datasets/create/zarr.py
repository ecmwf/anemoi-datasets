# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Any
from typing import Optional

import numpy as np
import zarr
from numpy.typing import NDArray

from anemoi.datasets import Zarr2AndZarr3

from .synchronise import NoSynchroniser
from .synchronise import Synchroniser


def add_zarr_dataset(
    *,
    name: str,
    dtype: np.dtype = None,
    fill_value: np.generic = None,
    zarr_root: zarr.Group,
    shape: tuple[int, ...] = None,
    array: NDArray[Any] = None,
    overwrite: bool = True,
    dimensions: tuple[str, ...] = None,
    **kwargs,
) -> zarr.Array:
    """Add a dataset to a Zarr group.

    Parameters
    ----------
    name : str
        Name of the dataset.
    dtype : np.dtype, optional
        Data type of the dataset.
    fill_value : np.generic, optional
        Fill value for the dataset.
    zarr_root : zarr.Group
        Root Zarr group.
    shape : tuple[int, ...], optional
        Shape of the dataset.
    array : NDArray[Any], optional
        Array to initialize the dataset with.
    overwrite : bool
        Whether to overwrite existing dataset.
    dimensions : tuple[str, ...]
        Dimensions of the dataset.
    **kwargs
        Additional arguments for Zarr dataset creation.

    Returns
    -------
    zarr.Array
        The created Zarr array.
    """
    assert dimensions is not None, "Please pass dimensions to add_zarr_dataset."
    assert isinstance(dimensions, (tuple, list))

    if dtype is None:
        assert array is not None, (name, shape, array, dtype, zarr_root)
        dtype = array.dtype

    if shape is None:
        assert array is not None, (name, shape, array, dtype, zarr_root)
        shape = array.shape

    if array is not None:
        array, dtype = Zarr2AndZarr3.cast_dtype_datetime64(array, dtype)

        assert array.shape == shape, (array.shape, shape)
        a = Zarr2AndZarr3.create_array(
            zarr_root,
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

    dtype = Zarr2AndZarr3.change_dtype_datetime64(dtype)
    a = Zarr2AndZarr3.create_array(
        zarr_root,
        name,
        shape=shape,
        dtype=dtype,
        overwrite=overwrite,
        **kwargs,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = dimensions
    return a


class ZarrBuiltRegistry:
    """A class to manage the creation and access of Zarr datasets."""

    name_lengths = "lengths"
    name_flags = "flags"
    lengths = None
    flags = None
    z = None

    def __init__(self, path: str, synchronizer_path: Optional[str] = None, use_threads: bool = False):
        """Initialize the ZarrBuiltRegistry.

        Parameters
        ----------
        path : str
            Path to the Zarr store.
        synchronizer_path : Optional[str], optional
            Path to the synchronizer.
        use_threads : bool
            Whether to use thread-based synchronization.
        """

        assert isinstance(path, str), path
        self.zarr_path = path

        self.synchronizer = Synchroniser(synchronizer_path) if synchronizer_path else NoSynchroniser()

    def clean(self) -> None:
        """Clean up the synchronizer path."""
        self.synchronizer.clean()

    def _open_write(self) -> zarr.Group:
        """Open the Zarr store in write mode."""
        return zarr.open(self.zarr_path, mode="r+")

    def _open_read(self, sync: bool = True) -> zarr.Group:
        """Open the Zarr store in read mode.

        Parameters
        ----------
        sync : bool
            Whether to use synchronization.

        Returns
        -------
        zarr.Group
            The opened Zarr group.
        """
        return zarr.open(self.zarr_path, mode="r")

    def new_dataset(self, *args, **kwargs) -> None:
        """Create a new dataset in the Zarr store.

        Parameters
        ----------
        *args
            Positional arguments for dataset creation.
        **kwargs
            Keyword arguments for dataset creation.
        """
        with self.synchronizer:
            z = self._open_write()
            zarr_root = z["_build"]
            add_zarr_dataset(*args, zarr_root=zarr_root, overwrite=True, dimensions=("tmp",), **kwargs)
            del z

    def add_to_history(self, action: str, **kwargs) -> None:
        """Add an action to the history attribute of the Zarr store.

        Parameters
        ----------
        action : str
            The action to record.
        **kwargs
            Additional information about the action.
        """
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

    def get_lengths(self) -> list[int]:
        """Get the lengths dataset.

        Returns
        -------
        list[int]
            The lengths dataset.
        """
        with self.synchronizer:
            z = self._open_read()
            lengths = list(z["_build"][self.name_lengths][:])
            del z
        return lengths

    def get_flags(self, **kwargs) -> list[bool]:
        """Get the flags dataset.

        Parameters
        ----------
        **kwargs
            Additional arguments for reading the dataset.

        Returns
        -------
        list[bool]
            The flags dataset.
        """
        with self.synchronizer:
            z = self._open_read(**kwargs)
            flags = list(z["_build"][self.name_flags][:])
            del z
        return flags

    def get_flag(self, i: int) -> bool:
        """Get a specific flag.

        Parameters
        ----------
        i : int
            Index of the flag.

        Returns
        -------
        bool
            The flag value.
        """
        with self.synchronizer:
            z = self._open_read()
            flag = z["_build"][self.name_flags][i]
            del z
        return flag

    def set_flag(self, i: int, value: bool = True) -> None:
        """Set a specific flag.

        Parameters
        ----------
        i : int
            Index of the flag.
        value : bool
            Value to set the flag to.
        """
        with self.synchronizer:
            z = self._open_write()
            z.attrs["latest_write_timestamp"] = (
                datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
            )
            z["_build"][self.name_flags][i] = value
            del z

    def ready(self) -> bool:
        """Check if all flags are set.

        Returns
        -------
        bool
            True if all flags are set, False otherwise.
        """
        return all(self.get_flags())

    def create(self, lengths: list[int], overwrite: bool = False) -> None:
        """Create the lengths and flags datasets.

        Parameters
        ----------
        lengths : list[int]
            Lengths to initialize the dataset with.
        overwrite : bool
            Whether to overwrite existing datasets.
        """
        self.new_dataset(name=self.name_lengths, array=np.array(lengths, dtype="i4"))
        self.new_dataset(name=self.name_flags, array=np.array([False] * len(lengths), dtype=bool))
        self.add_to_history("initialised")

    def reset(self, lengths: list[int]) -> None:
        """Reset the lengths and flags datasets.

        Parameters
        ----------
        lengths : list[int]
            Lengths to initialize the dataset with.
        """
        return self.create(lengths, overwrite=True)

    def add_provenance(self, name: str) -> None:
        """Add provenance information to the Zarr store.

        Parameters
        ----------
        name : str
            Name of the provenance attribute.
        """
        with self.synchronizer:
            z = self._open_write()

            if name in z.attrs:
                return

            from anemoi.utils.provenance import gather_provenance_info

            z.attrs[name] = gather_provenance_info()
            del z
