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
from functools import cached_property
from typing import Any

import numpy as np
import zarr
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets import MissingDateError
from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.dataset import Shape
from anemoi.datasets.usage.dataset import TupleIndex
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.debug import Source
from anemoi.datasets.usage.debug import debug_indexing
from anemoi.datasets.usage.gridded.indexing import expand_list_indexing

from ..store import ZarrStore

LOG = logging.getLogger(__name__)


class GriddedZarr(ZarrStore):
    """A zarr dataset."""

    def __init__(self, group: zarr.hierarchy.Group, path: str = None) -> None:
        super().__init__(group, path=path)
        self._missing = set()

    @property
    def missing(self) -> set[int]:
        """Return the missing dates of the dataset."""
        return self._missing

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
        return self.store.data.chunks

    @cached_property
    def shape(self) -> Shape:
        """Return the shape of the dataset."""
        return self.data.shape

    @cached_property
    def dtype(self) -> np.dtype:
        """Return the data type of the dataset."""
        return self.store.data.dtype

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Return the dates of the dataset."""
        return self.store.dates[:]  # Convert to numpy

    @property
    def latitudes(self) -> NDArray[Any]:
        """Return the latitudes of the dataset."""
        try:
            return self.store.latitudes[:]
        except AttributeError:
            LOG.warning("No 'latitudes' in %r, trying 'latitude'", self)
            return self.store.latitude[:]

    @property
    def longitudes(self) -> NDArray[Any]:
        """Return the longitudes of the dataset."""
        try:
            return self.store.longitudes[:]
        except AttributeError:
            LOG.warning("No 'longitudes' in %r, trying 'longitude'", self)
            return self.store.longitude[:]

    @property
    def statistics(self) -> dict[str, NDArray[Any]]:
        """Return the statistics of the dataset."""
        return dict(
            mean=self.store.mean[:],
            stdev=self.store.stdev[:],
            maximum=self.store.maximum[:],
            minimum=self.store.minimum[:],
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
            mean=self.store[func("mean")][:],
            stdev=self.store[func("stdev")][:],
            maximum=self.store[func("maximum")][:],
            minimum=self.store[func("minimum")][:],
        )

    @property
    def resolution(self) -> str:
        """Return the resolution of the dataset."""
        return self.store.attrs["resolution"]

    @property
    def field_shape(self) -> tuple:
        """Return the field shape of the dataset."""
        try:
            return tuple(self.store.attrs["field_shape"])
        except KeyError:
            LOG.warning("No 'field_shape' in %r, assuming 1D fields", self)
            return (self.shape[-1],)

    @property
    def frequency(self) -> datetime.timedelta:
        """Return the frequency of the dataset."""
        try:
            return frequency_to_timedelta(self.store.attrs["frequency"])
        except KeyError:
            LOG.warning("No 'frequency' in %r, computing from 'dates'", self)
        dates = self.dates
        return dates[1].astype(object) - dates[0].astype(object)

    @property
    def name_to_index(self) -> dict[str, int]:
        """Return the name to index mapping of the dataset."""
        if "variables" in self.store.attrs:
            return {n: i for i, n in enumerate(self.store.attrs["variables"])}
        return self.store.attrs["name_to_index"]

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
        result = self.store.attrs.get("constant_fields")
        if result is None:
            LOG.warning("No 'constant_fields' attribute in %r, computing them", self)
        return self.computed_constant_fields()

    @property
    def variables_metadata(self) -> dict[str, Any]:
        """Return the metadata of the variables."""
        return self.store.attrs.get("variables_metadata", {})

    def __repr__(self) -> str:
        """Return the string representation of the dataset."""
        return self.path

    def end_of_statistics_date(self) -> np.datetime64:
        """Return the end date of the statistics."""
        return self.dates[-1]

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        """Return the specific metadata of the dataset."""
        return super().metadata_specific(
            attrs=dict(self.store.attrs),
            chunks=self.chunks,
            dtype=str(self.dtype),
            path=self.path,
        )

    def source(self, index: int) -> Source:
        """Return the source of the dataset."""
        return Source(self, index, info=self.path)

    def mutate(self) -> Dataset:
        """Mutate the dataset if it has missing dates."""

        if len(self.store.attrs.get("missing_dates", [])):
            LOG.warning(f"Dataset {self} has missing dates")
            return ZarrWithMissingDates(self.store, self.path)
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

    @cached_property
    def origins(self):
        origins = self.store.attrs.get("origins")

        if origins is None:
            import rich

            rich.print(dict(self.store.attrs))
            raise ValueError(f"No 'origins' in {self.dataset_name}")

        # version = origins["version"]
        origins = origins["origins"]

        result = {}

        for origin in origins:
            for v in origin["variables"]:
                result[v] = origin["origin"]

        return result

    def project(self, projection):
        slices = tuple(slice(0, i, 1) for i in self.shape)
        return projection.from_store(slices, self).apply(projection)

    @property
    def dataset_name(self) -> str:
        """Return the name of the dataset."""
        return self.store.attrs.get("recipe", {}).get("name", self.path)

    def usage_factory_load(self, name):
        import importlib

        package, symbol = name.split(".")
        module = importlib.import_module(f".{package}", package=__package__)
        return getattr(module, symbol)


class ZarrWithMissingDates(GriddedZarr):
    """A zarr dataset with missing dates."""

    def __init__(self, store: zarr.hierarchy.Group, path: str) -> None:
        """Initialize the ZarrWithMissingDates dataset with a path or zarr group."""
        super().__init__(store, path)

        missing_dates = self.store.attrs.get("missing_dates", [])
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

    # def origin(self, index):
    #     if index[0] in self.missing:
    #         self._report_missing(index[0])
    #     return super().origin(index)
