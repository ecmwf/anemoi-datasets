# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import xarray as xr

LOG = logging.getLogger(__name__)


def patch_attributes(ds: xr.Dataset, attributes: dict[str, dict[str, Any]]) -> Any:
    """Patch the attributes of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    attributes : Dict[str, Dict[str, Any]]
        The attributes to patch.

    Returns
    -------
    Any
        The patched dataset.
    """
    for name, value in attributes.items():
        variable = ds[name]
        variable.attrs.update(value)

    return ds


def patch_coordinates(ds: xr.Dataset, coordinates: list[str]) -> Any:
    """Patch the coordinates of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    coordinates : List[str]
        The coordinates to patch.

    Returns
    -------
    Any
        The patched dataset.
    """
    for name in coordinates:
        ds = ds.assign_coords({name: ds[name]})

    return ds


def patch_rename(ds: xr.Dataset, renames: dict[str, str]) -> Any:
    """Rename variables in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    renames : dict[str, str]
        Mapping from old variable names to new variable names.

    Returns
    -------
    Any
        The patched dataset.
    """
    return ds.rename(renames)


def patch_sort_coordinate(ds: xr.Dataset, sort_coordinates: list[str]) -> Any:
    """Sort the coordinates of the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    sort_coordinates : List[str]
        The coordinates to sort.

    Returns
    -------
    Any
        The patched dataset.
    """

    for name in sort_coordinates:
        ds = ds.sortby(name)
    return ds


PATCHES = {
    "attributes": patch_attributes,
    "coordinates": patch_coordinates,
    "rename": patch_rename,
    "sort_coordinates": patch_sort_coordinate,
}


def patch_dataset(ds: xr.Dataset, patch: dict[str, dict[str, Any]]) -> Any:
    """Patch the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    patch : Dict[str, Dict[str, Any]]
        The patch to apply.

    Returns
    -------
    Any
        The patched dataset.
    """

    ORDER = ["coordinates", "attributes", "rename", "sort_coordinates"]
    for what, values in sorted(patch.items(), key=lambda x: ORDER.index(x[0])):
        if what not in PATCHES:
            raise ValueError(f"Unknown patch type {what!r}")

        ds = PATCHES[what](ds, values)

    return ds
