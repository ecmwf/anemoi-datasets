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


def patch_subset_dataset(ds, selection) -> Any:
    """Patch the dataset by selecting a subset with xarray sel.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    selection: dict
        Keys should be dimension names, values the selection to apply.

    Returns
    -------
    Any
        The patched dataset.
    """

    ds = ds.sel(selection)

    return ds


def patch_analysis_lead_to_valid_time(ds, time_coord_names) -> Any:
    """Patch the dataset by converting analysis/lead time to valid time.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    time_coord_names : dict[str, str]
        The names of the time coordinates. The keys must be:
        analysis_time_coordinate : str
            The name of the analysis time coordinate.
        lead_time_coordinate : str
            The name of the lead time coordinate.
        valid_time_coordinate : str
            The name of the valid time coordinate.

    Returns
    -------
    Any
        The patched dataset.
    """

    analysis_time_coordinate = time_coord_names["analysis_time_coordinate"]
    lead_time_coordinate = time_coord_names["lead_time_coordinate"]
    valid_time_coordinate = time_coord_names["valid_time_coordinate"]

    valid_time = ds[analysis_time_coordinate] + ds[lead_time_coordinate]

    ds = (
        ds.assign_coords({valid_time_coordinate: valid_time})
        .stack(time_index=[analysis_time_coordinate, lead_time_coordinate])
        .set_index(time_index=valid_time_coordinate)
        .rename(time_index=valid_time_coordinate)
        .drop_vars([analysis_time_coordinate, lead_time_coordinate])
    )

    return ds


def patch_rolling_sum(ds, vars_summation_period) -> Any:
    """Patch the dataset by converting analysis/lead time to valid time.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to patch.
    vars_summation_period: dict
        The variables and summation period. The keys must be:
        dim : str
            The dimension to apply the rolling sum.
        steps : int
            The number of steps to sum over.
        vars : list[str]
            The variables to apply the rolling sum.

    Returns
    -------
    Any
        The patched dataset.
    """
    dim = vars_summation_period["dim"]
    steps = vars_summation_period["steps"]
    vars = vars_summation_period["vars"]

    for var in vars:
        ds[var] = ds[var].rolling(dim={dim: steps}, min_periods=steps).sum()

    return ds


PATCHES = {
    "attributes": patch_attributes,
    "coordinates": patch_coordinates,
    "rename": patch_rename,
    "sort_coordinates": patch_sort_coordinate,
    "analysis_lead_to_valid_time": patch_analysis_lead_to_valid_time,
    "rolling_sum": patch_rolling_sum,
    "subset_dataset": patch_subset_dataset,
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

    ORDER = [
        "coordinates",
        "attributes",
        "rename",
        "sort_coordinates",
        "subset_dataset",
        "analysis_lead_to_valid_time",
        "rolling_sum",
    ]
    for what, values in sorted(patch.items(), key=lambda x: ORDER.index(x[0])):
        if what not in PATCHES:
            raise ValueError(f"Unknown patch type {what!r}")

        ds = PATCHES[what](ds, values)

    return ds
