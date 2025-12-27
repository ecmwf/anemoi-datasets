# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import os
import shutil
from functools import cached_property
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

from anemoi.datasets.usage.misc import as_first_date
from anemoi.datasets.usage.misc import as_last_date

from .statistics import default_statistics_dates

LOG = logging.getLogger(__name__)


class Synchronizer:
    """A placeholder for now"""

    def __init__(self, path):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def build_statistics_dates(
    dates: list[datetime.datetime],
    start: datetime.datetime | None,
    end: datetime.datetime | None,
) -> tuple[str, str]:
    """Compute the start and end dates for the statistics.

    Parameters
    ----------
    dates : list of datetime.datetime
        The list of dates.
    start : Optional[datetime.datetime]
        The start date.
    end : Optional[datetime.datetime]
        The end date.

    Returns
    -------
    tuple of str
        The start and end dates in ISO format.
    """
    # if not specified, use the default statistics dates
    default_start, default_end = default_statistics_dates(dates)
    if start is None:
        start = default_start
    if end is None:
        end = default_end

    # in any case, adapt to the actual dates in the dataset
    start = as_first_date(start, dates)
    end = as_last_date(end, dates)

    # and convert to datetime to isoformat
    start = start.astype(datetime.datetime)
    end = end.astype(datetime.datetime)
    return (start.isoformat(), end.isoformat())


class Dataset:
    """A class to represent a dataset."""

    def __init__(self, path: str, overwrite: bool = False, update: bool = False) -> None:
        self.path = path
        self.overwrite = overwrite
        self.update = update

        if self.overwrite and not self.update:
            raise ValueError("Cannot use overwrite without update")

        _, ext = os.path.splitext(self.path)
        if ext != ".zarr":
            raise ValueError(f"Unsupported extension={ext} for path={self.path}")

        if overwrite:
            try:
                shutil.rmtree(self.path)
            except FileNotFoundError:
                pass

        mode = "r"
        if overwrite or update:
            mode = "a" if update else "w"

        self.store = zarr.open(self.path, mode=mode)
        self.synchronizer = Synchronizer(self.path)

    def add_dataset(
        self,
        *,
        name: str,
        dtype: np.dtype = None,
        fill_value: np.generic = None,
        shape: tuple[int, ...] = None,
        array: NDArray[Any] = None,
        overwrite: bool = True,
        dimensions: tuple[str, ...] = None,
        chunks: tuple[int, ...] = None,
    ) -> zarr.Array:
        """Add a dataset to a Zarr group."""
        assert dimensions is not None, "Please pass dimensions to add_zarr_dataset."
        assert isinstance(dimensions, (tuple, list))

        bits = name.split("/")
        zarr_root = self.store
        for b in bits[:-1]:
            if b not in zarr_root:
                zarr_root = zarr_root.create_group(b)
            else:
                zarr_root = zarr_root[b]
        name = bits[-1]

        if dtype is None:
            assert array is not None, (name, shape, array, dtype, zarr_root)
            dtype = array.dtype

        if shape is None:
            assert array is not None, (name, shape, array, dtype, zarr_root)
            shape = array.shape

        if array is not None:
            assert array.shape == shape, (array.shape, shape)
            a = zarr_root.create_dataset(
                name,
                shape=shape,
                dtype=dtype,
                overwrite=overwrite,
                chunks=chunks,
            )
            a[...] = array
            a.attrs["_ARRAY_DIMENSIONS"] = dimensions
            return a

        if fill_value is None:
            if str(dtype).startswith("float") or str(dtype).startswith("numpy.float"):
                fill_value = np.nan
            elif str(dtype).startswith("datetime64") or str(dtype).startswith("numpy.datetime64"):
                fill_value = np.datetime64("NaT")
            # elif str(dtype).startswith("timedelta64") or str(dtype).startswith(
            #    "numpy.timedelta64"
            # ):
            #    kwargs["fill_value"] = np.timedelta64("NaT")
            elif str(dtype).startswith("int") or str(dtype).startswith("numpy.int"):
                fill_value = 0
            elif str(dtype).startswith("bool") or str(dtype).startswith("numpy.bool"):
                fill_value = False
            else:
                raise ValueError(f"No fill_value for dtype={dtype}")

        print(
            f"Creating zarr dataset {name} with shape={shape}, dtype={dtype}, fill_value={fill_value}, chunks={chunks}"
        )
        a = zarr_root.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            overwrite=overwrite,
            chunks=chunks,
            fill_value=fill_value,
        )
        a.attrs["_ARRAY_DIMENSIONS"] = dimensions
        return a

    def update_metadata(self, *args, **kwargs) -> None:
        """Update the metadata of the dataset."""

        metadata = dict(*args, **kwargs)
        metadata = json.loads(json.dumps(metadata, default=str))

        self.store.attrs.update(metadata)

    # def get_zarr_chunks(self) -> tuple:
    #     """Get the chunks of the Zarr dataset.

    #     Returns
    #     -------
    #     tuple
    #         The chunks of the Zarr dataset.
    #     """
    #     import zarr

    #     z = zarr.open(self.path, mode="r")
    #     return z["data"].chunks

    @staticmethod
    def check_name(
        resolution: str,
        dates: list[datetime.datetime],
        frequency: datetime.timedelta,
        raise_exception: bool = True,
    ) -> None:
        """Check the name of the dataset."""
        LOG.warning("BACK: Dataset.check_name is not implemented fully yet.")
        return
        # basename, _ = os.path.splitext(os.path.basename(self.path))
        # try:
        #     DatasetName(basename, resolution, dates[0], dates[-1], frequency).raise_if_not_valid()
        # except Exception as e:
        #     if raise_exception and not is_test:
        #         raise e
        #     else:
        #         LOG.warning(f"Dataset name error: {e}")

    def clean(self) -> None:
        """Clean up the synchronizer path."""
        if self.synchronizer_path is not None:
            try:
                shutil.rmtree(self.synchronizer_path)
            except FileNotFoundError:
                pass

        _build = self.zarr_path + "/_build"
        try:
            shutil.rmtree(_build)
        except FileNotFoundError:
            pass

    # def get_lengths(self) -> list[int]:
    #     """Get the lengths dataset.

    #     Returns
    #     -------
    #     list[int]
    #         The lengths dataset.
    #     """
    #     z = self._open_read()
    #     return list(z["_build"][self.name_lengths][:])

    def flags(self) -> list[bool]:
        return self.store["_build"]["flags"][:]

    def flag(self, i: int, value: bool | None = None) -> bool:
        flags = self.store["_build"]["flags"]
        if value is not None:
            with self.synchronizer:
                flags[i] = value

        return flags[i]

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

    # def reset(self, lengths: list[int]) -> None:
    #     """Reset the lengths and flags datasets.

    #     Parameters
    #     ----------
    #     lengths : list[int]
    #         Lengths to initialize the dataset with.
    #     """
    #     return self.create(lengths, overwrite=True)

    def add_provenance(self, name: str) -> None:

        from anemoi.utils.provenance import gather_provenance_info

        if name not in self.store.attrs:
            self.store.attrs[name] = gather_provenance_info()

    def init_progress(self, lengths: tuple[int, ...]) -> None:
        """Initialize the progress tracking datasets.

        Parameters
        ----------
        lengths : tuple[int, ...]
            The lengths of each group.
        """
        LOG.warning("TODO: Initializing progress tracking datasets.")

        self.add_dataset(
            name="_build/lengths",
            array=np.array(lengths, dtype="i4"),
            overwrite=self.overwrite,
            dimensions=("group",),
        )
        self.add_dataset(
            name="_build/flags",
            array=np.array([False] * len(lengths), dtype=bool),
            overwrite=self.overwrite,
            dimensions=("group",),
        )
        # self.create(lengths=lengths, overwrite=self.update)

    @cached_property
    def dates(self):
        return self.store["dates"][:]

    @property
    def data(self):
        return self.store["data"]
