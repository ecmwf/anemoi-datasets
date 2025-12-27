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

from anemoi.datasets.usage.misc import as_first_date
from anemoi.datasets.usage.misc import as_last_date

from .gridded.statistics import default_statistics_dates

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

    def add_array(
        self,
        *,
        name: str,
        dimensions: tuple[str, ...],
        **kwargs: Any,
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

        #################

        kwargs.setdefault("override", True)

        #################

        LOG.info(f"Creating array {name} with kwargs={kwargs} (dimensions={dimensions})")

        a = zarr_root.create_dataset(name, **kwargs)
        a.attrs["_ARRAY_DIMENSIONS"] = dimensions
        return a

    def update_metadata(self, *args, **kwargs) -> None:
        """Update the metadata of the dataset."""

        metadata = dict(*args, **kwargs)
        metadata = json.loads(json.dumps(metadata, default=str))

        with self.synchronizer:
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

    ##################################
    # Progress tracking methods
    ##################################

    def total_todo(self) -> int:
        return len(self.store["_build"]["flags"])

    def todo_remaining(self) -> int:
        with self.synchronizer:
            flags = self.store["_build"]["flags"]
            return np.sum(~flags)

    def mark_done(self, i: int):
        with self.synchronizer:
            flags = self.store["_build"]["flags"]
            flags[i] = True

    def is_done(self, i: int) -> bool:
        with self.synchronizer:
            flags = self.store["_build"]["flags"]
            return flags[i]

    def ready(self) -> bool:
        """Check if all flags are set."""
        return self.todo_remaining() == 0

    ##################################

    def xxxxxcreate(self, lengths: list[int], overwrite: bool = False) -> None:
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

    def initalise_done_flags(self, groups: int) -> None:
        """Initialize the progress tracking datasets."""

        self.add_array(
            name="_build/flags",
            data=np.array([False] * groups, dtype=bool),
            dimensions=("group",),
        )

    def remove_group(self, name: str) -> None:
        """Remove a group from the dataset.

        Parameters
        ----------
        name : str
            The name of the group to remove.
        """

        if name in self.store:
            LOG.info(f"Removing group {name} from dataset")
            del self.store[name]

    @cached_property
    def dates(self):
        return self.store["dates"][:]

    @property
    def data(self):
        return self.store["data"]
