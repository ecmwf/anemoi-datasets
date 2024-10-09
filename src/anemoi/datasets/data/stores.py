# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
import warnings
from functools import cached_property
from urllib.parse import urlparse

import numpy as np
import zarr
from anemoi.utils.dates import frequency_to_timedelta

from . import MissingDateError
from .dataset import Dataset
from .debug import DEBUG_ZARR_LOADING
from .debug import Node
from .debug import Source
from .debug import debug_indexing
from .indexing import expand_list_indexing
from .misc import load_config

LOG = logging.getLogger(__name__)


class ReadOnlyStore(zarr.storage.BaseStore):
    def __delitem__(self, key):
        raise NotImplementedError()

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()


class HTTPStore(ReadOnlyStore):
    """We write our own HTTPStore because the one used by zarr (s3fs)
    does not play well with fork() and multiprocessing.
    """

    def __init__(self, url):
        self.url = url

    def __getitem__(self, key):
        import requests

        r = requests.get(self.url + "/" + key)

        if r.status_code == 404:
            raise KeyError(key)

        r.raise_for_status()
        return r.content


class S3Store(ReadOnlyStore):
    """We write our own S3Store because the one used by zarr (s3fs)
    does not play well with fork(). We also get to control the s3 client
    options using the anemoi configs.
    """

    def __init__(self, url, region=None):
        from anemoi.utils.s3 import s3_client

        _, _, self.bucket, self.key = url.split("/", 3)
        self.s3 = s3_client(self.bucket, region=region)

    def __getitem__(self, key):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key + "/" + key)
        except self.s3.exceptions.NoSuchKey:
            raise KeyError(key)

        return response["Body"].read()


class DebugStore(ReadOnlyStore):
    """A store to debug the zarr loading."""

    def __init__(self, store):
        assert not isinstance(store, DebugStore)
        self.store = store

    def __getitem__(self, key):
        # print()
        print("GET", key, self)
        # traceback.print_stack(file=sys.stdout)
        return self.store[key]

    def __len__(self):
        return len(self.store)

    def __iter__(self):
        warnings.warn("DebugStore: iterating over the store")
        return iter(self.store)

    def __contains__(self, key):
        return key in self.store


def name_to_zarr_store(path_or_url):
    store = path_or_url

    if store.startswith("s3://"):
        store = S3Store(store)

    elif store.startswith("http://") or store.startswith("https://"):
        parsed = urlparse(store)
        bits = parsed.netloc.split(".")
        if len(bits) == 5 and (bits[1], bits[3], bits[4]) == ("s3", "amazonaws", "com"):
            s3_url = f"s3://{bits[0]}{parsed.path}"
            store = S3Store(s3_url, region=bits[2])
        else:
            store = HTTPStore(store)

    return store


def open_zarr(path, dont_fail=False, cache=None):
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

        return zarr.convenience.open(store, "r")
    except zarr.errors.PathNotFoundError:
        if not dont_fail:
            raise zarr.errors.PathNotFoundError(path)


class Zarr(Dataset):
    """A zarr dataset."""

    def __init__(self, path):
        if isinstance(path, zarr.hierarchy.Group):
            self.was_zarr = True
            self.path = str(id(path))
            self.z = path
        else:
            self.was_zarr = False
            self.path = str(path)
            self.z = open_zarr(self.path)

        # This seems to speed up the reading of the data a lot
        self.data = self.z.data
        self.missing = set()

    @classmethod
    def from_name(cls, name):
        if name.endswith(".zip") or name.endswith(".zarr"):
            return Zarr(name)
        return Zarr(zarr_lookup(name))

    def __len__(self):
        return self.data.shape[0]

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n):
        return self.data[n]

    def _unwind(self, index, rest, shape, axis, axes):
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
    def chunks(self):
        return self.z.data.chunks

    @cached_property
    def shape(self):
        return self.data.shape

    @cached_property
    def dtype(self):
        return self.z.data.dtype

    @cached_property
    def dates(self):
        return self.z.dates[:]  # Convert to numpy

    @property
    def latitudes(self):
        try:
            return self.z.latitudes[:]
        except AttributeError:
            LOG.warning("No 'latitudes' in %r, trying 'latitude'", self)
            return self.z.latitude[:]

    @property
    def longitudes(self):
        try:
            return self.z.longitudes[:]
        except AttributeError:
            LOG.warning("No 'longitudes' in %r, trying 'longitude'", self)
            return self.z.longitude[:]

    @property
    def statistics(self):
        return dict(
            mean=self.z.mean[:],
            stdev=self.z.stdev[:],
            maximum=self.z.maximum[:],
            minimum=self.z.minimum[:],
        )

    def statistics_tendencies(self, delta=None):
        if delta is None:
            delta = self.frequency
        if isinstance(delta, int):
            delta = f"{delta}h"
        from anemoi.utils.dates import frequency_to_string
        from anemoi.utils.dates import frequency_to_timedelta

        delta = frequency_to_timedelta(delta)
        delta = frequency_to_string(delta)

        def func(k):
            return f"statistics_tendencies_{delta}_{k}"

        return dict(
            mean=self.z[func("mean")][:],
            stdev=self.z[func("stdev")][:],
            maximum=self.z[func("maximum")][:],
            minimum=self.z[func("minimum")][:],
        )

    @property
    def resolution(self):
        return self.z.attrs["resolution"]

    @property
    def field_shape(self):
        try:
            return tuple(self.z.attrs["field_shape"])
        except KeyError:
            LOG.warning("No 'field_shape' in %r, assuming 1D fields", self)
            return (self.shape[-1],)

    @property
    def frequency(self):
        try:
            return frequency_to_timedelta(self.z.attrs["frequency"])
        except KeyError:
            LOG.warning("No 'frequency' in %r, computing from 'dates'", self)
        dates = self.dates
        return dates[1].astype(object) - dates[0].astype(object)

    @property
    def name_to_index(self):
        if "variables" in self.z.attrs:
            return {n: i for i, n in enumerate(self.z.attrs["variables"])}
        return self.z.attrs["name_to_index"]

    @property
    def variables(self):
        return [
            k
            for k, v in sorted(
                self.name_to_index.items(),
                key=lambda x: x[1],
            )
        ]

    @property
    def variables_metadata(self):
        return self.z.attrs.get("variables_metadata", {})

    def __repr__(self):
        return self.path

    def end_of_statistics_date(self):
        return self.dates[-1]

    def metadata_specific(self):
        return super().metadata_specific(
            attrs=dict(self.z.attrs),
            chunks=self.chunks,
            dtype=str(self.dtype),
        )

    def source(self, index):
        return Source(self, index, info=self.path)

    def mutate(self):
        if len(self.z.attrs.get("missing_dates", [])):
            LOG.warning(f"Dataset {self} has missing dates")
            return ZarrWithMissingDates(self.z if self.was_zarr else self.path)
        return self

    def tree(self):
        return Node(self, [], path=self.path)

    def get_dataset_names(self, names):
        name, _ = os.path.splitext(os.path.basename(self.path))
        names.add(name)


class ZarrWithMissingDates(Zarr):
    """A zarr dataset with missing dates."""

    def __init__(self, path):
        super().__init__(path)

        missing_dates = self.z.attrs.get("missing_dates", [])
        missing_dates = set([np.datetime64(x) for x in missing_dates])
        self.missing_to_dates = {i: d for i, d in enumerate(self.dates) if d in missing_dates}
        self.missing = set(self.missing_to_dates)

    def mutate(self):
        return self

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n):
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

        raise TypeError(f"Unsupported index {n} {type(n)}")

    def _report_missing(self, n):
        raise MissingDateError(f"Date {self.missing_to_dates[n]} is missing (index={n})")

    def tree(self):
        return Node(self, [], path=self.path, missing=sorted(self.missing))

    @property
    def label(self):
        return "zarr*"


def zarr_lookup(name, fail=True):

    if name.endswith(".zarr") or name.endswith(".zip"):
        return name

    config = load_config()["datasets"]

    if name in config["named"]:
        return config["named"][name]

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
                return full
        except zarr.errors.PathNotFoundError:
            pass

    if fail:
        raise ValueError(f"Cannot find a dataset that matched '{name}'. Tried: {tried}")

    return None
