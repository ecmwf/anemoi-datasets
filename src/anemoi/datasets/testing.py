# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# A collection of functions to support pytest testing

import logging
from typing import Any

import numpy as np
import zarr

LOG = logging.getLogger(__name__)


def assert_field_list(
    fs: list[Any],
    size: int | None = None,
    start: Any | None = None,
    end: Any | None = None,
    constant: bool = False,
    skip: Any | None = None,
) -> None:
    """Asserts various properties of a list of fields.

    Parameters
    ----------
    fs : List[Any]
        List of fields to be checked.
    size : Optional[int], optional
        Expected size of the list. If None, the list must be non-empty.
    start : Optional[Any], optional
        Expected start metadata value. If None, no check is performed.
    end : Optional[Any], optional
        Expected end metadata value. If None, no check is performed.
    constant : bool, optional
        If True, checks that all fields are constant.
    skip : Optional[Any], optional
        Placeholder for future use.
    """
    import numpy as np

    if size is None:
        assert len(fs) > 0, fs
    else:
        assert len(fs) == size, (len(fs), size)

    first = fs[0]
    last = fs[-1]

    if constant:
        # TODO: add a check for constant fields
        pass
    else:
        assert start is None or first.metadata("valid_datetime") == start, (first.metadata("valid_datetime"), start)
        assert end is None or last.metadata("valid_datetime") == end, (last.metadata("valid_datetime"), end)
        print(first.datetime())

    print(last.metadata())

    first = first
    latitudes, longitudes = first.grid_points()

    assert len(latitudes.shape) == 1, latitudes.shape
    assert len(longitudes.shape) == 1, longitudes.shape

    assert len(latitudes) == len(longitudes), (len(latitudes), len(longitudes))
    data = first.to_numpy(flatten=True)

    assert len(data) == len(latitudes), (len(data), len(latitudes))

    north = np.max(latitudes)
    south = np.min(latitudes)
    east = np.max(longitudes)
    west = np.min(longitudes)

    assert north >= south, (north, south)
    assert east >= west, (east, west)
    assert north <= 90, north
    assert south >= -90, south
    assert east <= 360, east
    assert west >= -180, west


class IndexTester:
    """Class to test indexing of datasets."""

    def __init__(self, ds: Any) -> None:
        """Initialise the IndexTester.

        Parameters
        ----------
        ds : Any
            Dataset.
        """
        self.ds = ds
        self.np = ds[:]  # Numpy array

        assert self.ds.shape == self.np.shape, (self.ds.shape, self.np.shape)
        assert (self.ds == self.np).all()

    def __getitem__(self, index: Any) -> None:
        """Test indexing.

        Parameters
        ----------
        index : Any
            Index.
        """
        LOG.info("IndexTester: %s", index)
        if self.ds[index] is None:
            assert False, (self.ds, index)

        if not (self.ds[index] == self.np[index]).all():
            assert (self.ds[index] == self.np[index]).all()


def default_test_indexing(ds):

    t = IndexTester(ds)

    t[0:10, :, 0]
    t[:, 0:3, 0]
    # t[:, :, 0]
    t[0:10, 0:3, 0]
    t[:, :, :]

    if ds.shape[1] > 2:  # Variable dimension
        t[:, (1, 2), :]
        t[:, (1, 2)]

    t[0]
    t[0, :]
    t[0, 0, :]
    t[0, 0, 0, :]

    if ds.shape[2] > 1:  # Ensemble dimension
        t[0:10, :, (0, 1)]

    for i in range(3):
        t[i]
        start = 5 * i
        end = len(ds) - 5 * i
        step = len(ds) // 10

        t[start:end:step]
        t[start:end]
        t[start:]
        t[:end]
        t[::step]


class Trace:

    def __init__(self, ds):
        self.ds = ds
        self.f = open("trace.txt", "a")

    def __getattr__(self, name: str) -> Any:

        print(name, file=self.f, flush=True)
        return getattr(self.ds, name)

    def __len__(self) -> int:
        print("__len__", file=self.f, flush=True)
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        print("__getitem__", file=self.f, flush=True)
        return self.ds[index]


class FastArray:
    """A numpy-backed wrapper for zarr arrays that provides fast indexing."""

    def __init__(self, zarr_array):
        self._zarr = zarr_array
        self._data = np.array(zarr_array[:])

    def __getitem__(self, key):
        """Fast numpy-based indexing."""
        return self._data[key]

    def __array__(self, dtype=None):
        if dtype is None:
            return self._data
        return self._data.astype(dtype)

    def __getattr__(self, name):
        return getattr(self._zarr, name)


class FastGroup:
    """A wrapper for zarr groups that provides FastArray for all arrays."""

    def __init__(self, zarr_group):
        self._group = zarr_group
        self._arrays = {}

        for name in self._group.keys():
            item = self._group[name]
            if isinstance(item, zarr.Array):
                self._arrays[name] = FastArray(item)

    def __getitem__(self, key):
        if key in self._arrays:
            return self._arrays[key]

        item = self._group[key]
        if isinstance(item, zarr.Array):
            self._arrays[key] = FastArray(item)
            return self._arrays[key]
        elif isinstance(item, zarr.Group):
            return FastGroup(item)
        return item

    def __getattr__(self, name):
        return getattr(self._group, name)
