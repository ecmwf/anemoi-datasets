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
import os
import tempfile
from functools import cached_property
from typing import Any
from urllib.parse import urlparse

import numpy as np
import zarr
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from . import MissingDateError
from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .debug import DEBUG_ZARR_LOADING
from .debug import Node
from .debug import Source
from .debug import debug_indexing
from .indexing import expand_list_indexing
from .misc import load_config

LOG = logging.getLogger(__name__)


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


def name_to_zarr_store(path_or_url: str) -> Any:
    """Convert a path or URL to a zarr store."""
    store = path_or_url

    if store.startswith("s3://"):
        return S3Store(store)

    if store.startswith("http://") or store.startswith("https://"):

        if store.endswith(".zip"):
            import multiurl

            parsed = urlparse(store)

            # Zarr cannot handle zip files over HTTP
            tmpdir = tempfile.gettempdir()
            name = os.path.basename(parsed.path)
            path = os.path.join(tmpdir, name)
            LOG.warning("Zarr does not support zip files over HTTP, downloading to %s", path)
            if os.path.exists(path):
                LOG.warning("File %s already exists, reusing it", path)
                return name_to_zarr_store(path)

            LOG.warning("Downloading %s", store)

            multiurl.download(store, path + ".tmp")
            os.rename(path + ".tmp", path)
            return name_to_zarr_store(path)

        return HTTPStore(store)

    return store


def open_zarr(path: str, dont_fail: bool = False, cache: int = None) -> zarr.Group:
    """Open a zarr store from a path."""
    try:
        store = name_to_zarr_store(path)

        if DEBUG_ZARR_LOADING:
            if isinstance(store, str):
                import os

                if not os.path.isdir(store):
                    raise NotImplementedError(
                        "DEBUG_ZARR_LOADING is only implemented for DirectoryStore. "
                        "Please disable it for other backends."
                    )
                store = zarr.storage.DirectoryStore(store)
            store = DebugStore(store)

        if cache is not None:
            store = zarr.LRUStoreCache(store, max_size=cache)

        return zarr.open(store, mode="r")
    except FileNotFoundError:
        if not dont_fail:
            raise FileNotFoundError(f"Zarr store not found: {path}")


class Zarr(Dataset):
    """A zarr dataset."""

    def __init__(self, path: str | zarr.Group) -> None:
        """Initialize the Zarr dataset with a path or zarr group."""
        if isinstance(path, zarr.Group):
            self.was_zarr = True
            self.path = str(id(path))
            self.z = path
        else:
            self.was_zarr = False
            self.path = str(path)
            self.z = open_zarr(self.path)

        # This seems to speed up the reading of the data a lot
        self.data = self.z["data"]
        self._missing = set()

    @property
    def missing(self) -> set[int]:
        """Return the missing dates of the dataset."""
        return self._missing

    @classmethod
    def from_name(cls, name: str) -> "Zarr":
        """Create a Zarr dataset from a name."""
        if name.endswith(".zip") or name.endswith(".zarr"):
            return Zarr(name)
        return Zarr(zarr_lookup(name))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.data.shape[0]

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Retrieve an item from the dataset."""
        return self.data[n]

    def _unwind(self, index: int | slice | list | tuple, rest: list, shape: tuple, axis: int, axes: list) -> iter:
        """Unwind the index for multi-dimensional indexing."""
        if not isinstance(index, (int, slice, list, tuple)):
            try:
                # NumPy arrays, TensorFlow tensors, etc.
                index = tuple(index.tolist())
                assert not isinstance(index, bool), "Mask not supported"
            except AttributeError:
                pass

        if isinstance(index, (list, tuple)):
            axes.append(axis)  # Dimension of the concatenation
            for i in index:
                yield from self._unwind((slice(i, i + 1),), rest, shape, axis, axes)
            return

        if len(rest) == 0:
            yield (index,)
            return

        for n in self._unwind(rest[0], rest[1:], shape, axis + 1, axes):
            yield (index,) + n

    @cached_property
    def chunks(self) -> TupleIndex:
        """Return the chunks of the dataset."""
        return self.data.chunks

    @cached_property
    def shape(self) -> Shape:
        """Return the shape of the dataset."""
        return self.data.shape

    @cached_property
    def dtype(self) -> np.dtype:
        """Return the data type of the dataset."""
        return self.data.dtype

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Return the dates of the dataset."""
        return self.z["dates"][:]  # Convert to numpy

    @property
    def latitudes(self) -> NDArray[Any]:
        """Return the latitudes of the dataset."""
        return self.z["latitudes"][:]

    @property
    def longitudes(self) -> NDArray[Any]:
        """Return the longitudes of the dataset."""
        return self.z["longitudes"][:]

    @property
    def statistics(self) -> dict[str, NDArray[Any]]:
        """Return the statistics of the dataset."""
        return dict(
            mean=self.z["mean"][:],
            stdev=self.z["stdev"][:],
            maximum=self.z["maximum"][:],
            minimum=self.z["minimum"][:],
        )

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
        """Return the statistical tendencies of the dataset."""
        if delta is None:
            delta = self.frequency
        if isinstance(delta, int):
            delta = f"{delta}h"
        from anemoi.utils.dates import frequency_to_string
        from anemoi.utils.dates import frequency_to_timedelta

        delta = frequency_to_timedelta(delta)
        delta = frequency_to_string(delta)

        def func(k: str) -> str:
            return f"statistics_tendencies_{delta}_{k}"

        return dict(
            mean=self.z[func("mean")][:],
            stdev=self.z[func("stdev")][:],
            maximum=self.z[func("maximum")][:],
            minimum=self.z[func("minimum")][:],
        )

    @property
    def resolution(self) -> str:
        """Return the resolution of the dataset."""
        return self.z.attrs["resolution"]

    @property
    def field_shape(self) -> tuple:
        """Return the field shape of the dataset."""
        try:
            return tuple(self.z.attrs["field_shape"])
        except KeyError:
            LOG.warning("No 'field_shape' in %r, assuming 1D fields", self)
            return (self.shape[-1],)

    @property
    def frequency(self) -> datetime.timedelta:
        """Return the frequency of the dataset."""
        try:
            return frequency_to_timedelta(self.z.attrs["frequency"])
        except KeyError:
            LOG.warning("No 'frequency' in %r, computing from 'dates'", self)
        dates = self.dates
        return dates[1].astype(object) - dates[0].astype(object)

    @property
    def name_to_index(self) -> dict[str, int]:
        """Return the name to index mapping of the dataset."""
        if "variables" in self.z.attrs:
            return {n: i for i, n in enumerate(self.z.attrs["variables"])}
        return self.z.attrs["name_to_index"]

    @property
    def variables(self) -> list[str]:
        """Return the variables of the dataset."""
        return [
            k
            for k, v in sorted(
                self.name_to_index.items(),
                key=lambda x: x[1],
            )
        ]

    @cached_property
    def constant_fields(self) -> list[str]:
        """Return the constant fields of the dataset."""
        result = self.z.attrs.get("constant_fields")
        if result is None:
            LOG.warning("No 'constant_fields' attribute in %r, computing them", self)
        return self.computed_constant_fields()

    @property
    def variables_metadata(self) -> dict[str, Any]:
        """Return the metadata of the variables."""
        return self.z.attrs.get("variables_metadata", {})

    def __repr__(self) -> str:
        """Return the string representation of the dataset."""
        return self.path

    def end_of_statistics_date(self) -> np.datetime64:
        """Return the end date of the statistics."""
        return self.dates[-1]

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        """Return the specific metadata of the dataset."""
        return super().metadata_specific(
            attrs=dict(self.z.attrs),
            chunks=self.chunks,
            dtype=str(self.dtype),
            path=self.path,
        )

    def source(self, index: int) -> Source:
        """Return the source of the dataset."""
        return Source(self, index, info=self.path)

    def mutate(self) -> Dataset:
        """Mutate the dataset if it has missing dates."""
        if len(self.z.attrs.get("missing_dates", [])):
            LOG.warning(f"Dataset {self} has missing dates")
            return ZarrWithMissingDates(self.z if self.was_zarr else self.path)
        return self

    def tree(self) -> Node:
        """Return the tree representation of the dataset."""
        return Node(self, [], path=self.path)

    def get_dataset_names(self, names: set[str]) -> None:
        """Get the names of the datasets."""
        name, _ = os.path.splitext(os.path.basename(self.path))
        names.add(name)

    def collect_supporting_arrays(self, collected: set, *path: str) -> None:
        """Collect supporting arrays."""
        pass

    def collect_input_sources(self, collected: set) -> None:
        """Collect input sources."""
        pass


class ZarrWithMissingDates(Zarr):
    """A zarr dataset with missing dates."""

    def __init__(self, path: str | zarr.Group) -> None:
        """Initialize the ZarrWithMissingDates dataset with a path or zarr group."""
        super().__init__(path)

        missing_dates = self.z.attrs.get("missing_dates", [])
        missing_dates = {np.datetime64(x, "s") for x in missing_dates}
        self.missing_to_dates = {i: d for i, d in enumerate(self.dates) if d in missing_dates}
        self._missing = set(self.missing_to_dates)

    @property
    def missing(self) -> set[int]:
        """Return the missing dates of the dataset."""
        return self._missing

    def mutate(self) -> Dataset:
        """Mutate the dataset."""
        return self

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Retrieve an item from the dataset."""
        if isinstance(n, int):
            if n in self.missing:
                self._report_missing(n)
            return self.data[n]

        if isinstance(n, slice):
            common = set(range(*n.indices(len(self)))) & self.missing
            if common:
                self._report_missing(list(common)[0])
            return self.data[n]

        if isinstance(n, tuple):
            first = n[0]
            if isinstance(first, int):
                if first in self.missing:
                    self._report_missing(first)
                return self.data[n]

            if isinstance(first, slice):
                common = set(range(*first.indices(len(self)))) & self.missing
                if common:
                    self._report_missing(list(common)[0])
                return self.data[n]

            if isinstance(first, (list, tuple)):
                common = set(first) & self.missing
                if common:
                    self._report_missing(list(common)[0])
                return self.data[n]

            raise TypeError(f"Unsupported index {n} {type(n)}, {first} {type(first)}")

        raise TypeError(f"Unsupported index {n} {type(n)}")

    def _report_missing(self, n: int) -> None:
        """Report a missing date."""
        raise MissingDateError(f"Date {self.missing_to_dates[n]} is missing (index={n})")

    def tree(self) -> Node:
        """Return the tree representation of the dataset."""
        return Node(self, [], path=self.path, missing=sorted(self.missing))

    @property
    def label(self) -> str:
        """Return the label of the dataset."""
        return "zarr*"


QUIET = set()


def zarr_lookup(name: str, fail: bool = True) -> str | None:
    """Look up a zarr dataset by name."""

    config = load_config()["datasets"]
    use_search_path_not_found = config.get("use_search_path_not_found", False)

    if name.endswith(".zarr/"):
        LOG.warning("Removing trailing slash from path: %s", name)
        name = name[:-1]

    if name.endswith(".zarr") or name.endswith(".zip"):

        if os.path.exists(name):
            return name

        if not use_search_path_not_found:
            # There will be an error triggered by the open_zarr
            return name

        LOG.warning("File %s not found, trying to search in the search path", name)
        name = os.path.splitext(os.path.basename(name))[0]

    if name in config["named"]:
        if name not in QUIET:
            LOG.info("Opening `%s` as `%s`", name, config["named"][name])
            QUIET.add(name)
        return str(config["named"][name])

    tried = []
    for location in config["path"]:
        if not location.endswith("/"):
            location += "/"
        full = location + name + ".zarr"
        tried.append(full)
        try:
            z = open_zarr(full, dont_fail=True)
            if z is not None:
                # Cache for next time
                config["named"][name] = full
                if name not in QUIET:
                    LOG.info("Opening `%s` as `%s`", name, full)
                    QUIET.add(name)
                return full
        except FileNotFoundError:
            pass

    if fail:
        raise ValueError(f"Cannot find a dataset that matched '{name}'. Tried: {tried}")

    return None
