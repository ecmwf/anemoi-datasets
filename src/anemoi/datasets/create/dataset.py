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
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOG = logging.getLogger(__name__)


def _tidy_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable formats."""

    match obj:
        case datetime.datetime():
            return obj.isoformat()

        case np.datetime64():
            return obj.astype(object).isoformat()

        case datetime.timedelta():
            return frequency_to_string(obj)

        case _:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class Synchronizer:
    """A placeholder for now"""

    def __init__(self, path):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class Dataset:
    """A class to represent a dataset."""

    def __init__(self, path: str, overwrite: bool = False, update: bool = False, create: bool = False) -> None:

        if path.endswith("/"):
            path = path[:-1]

        self.path = path

        if overwrite and not create:
            raise ValueError("Cannot use overwrite without create")

        if create and update:
            raise ValueError("Cannot use create and update together")

        _, ext = os.path.splitext(self.path)
        if ext != ".zarr":
            raise ValueError(f"Unsupported extension={ext} for path={self.path}")

        if create:
            if os.path.exists(path):
                if overwrite or True:
                    LOG.warning(f"Overwriting existing dataset at '{path}'.")
                    shutil.rmtree(path)
                else:
                    raise FileExistsError(f"Dataset path '{path}' already exists. Use --overwrite to overwrite.")

        mode = "r"

        if create:
            mode = "w"

        if update:
            mode = "a"

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

        def _(d):
            if "data" in d:
                d = d.copy()
                d["data"] = f'[... {d["data"].shape} {d["data"].dtype} array ...]'
            return d

        LOG.info(f"Creating array {name} with kwargs={_(kwargs)} (dimensions={dimensions})")

        a = zarr_root.create_dataset(name, **kwargs)
        a.attrs["_ARRAY_DIMENSIONS"] = dimensions
        return a

    def update_metadata(self, *args, **kwargs) -> None:
        """Update the metadata of the dataset."""

        metadata = dict(*args, **kwargs)
        metadata = json.loads(json.dumps(metadata, default=_tidy_json))

        with self.synchronizer:
            self.store.attrs.update(metadata)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the dataset."""
        with self.synchronizer:
            return self.store.attrs.get(key, default)

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

    def group_to_range(self, group: int) -> tuple[int, int]:
        """Convert group indices to slice in the data array."""

        with self.synchronizer:
            lengths = self.store["_build"]["lengths"][:]
            start = sum(lengths[:group])
            end = start + lengths[group]
            return (start, end)

    ##################################

    def add_provenance(self, name: str) -> None:

        from anemoi.utils.provenance import gather_provenance_info

        if name not in self.store.attrs:
            self.store.attrs[name] = gather_provenance_info()

    # For statistics about
    def initalise_groups_lengths(self, lengths: list[int]) -> None:
        """Initialize the progress tracking datasets."""

        self.add_array(
            name="_build/lengths",
            data=np.array(lengths, dtype=np.int64),
            dimensions=("group",),
        )

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

    @cached_property
    def frequency(self) -> datetime.timedelta | None:
        """Get the dataset frequency."""
        frequency = self.store.attrs.get("frequency")
        return frequency_to_timedelta(frequency) if frequency is not None else None

    def touch(self) -> None:
        self.update_metadata(latest_write_timestamp=datetime.datetime.now(datetime.UTC).replace(tzinfo=None))
