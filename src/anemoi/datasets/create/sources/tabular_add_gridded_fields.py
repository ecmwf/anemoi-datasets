# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tabular filter that samples gridded fields at the observation locations.

The module lives with the sources so that it is loaded (and the filter
registered in the anemoi-transform filter registry) before the recipe is
parsed, see ``anemoi.datasets.create.recipe.action._factories``.
"""

import functools
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


def _xyz(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Convert latitudes and longitudes in degrees to points on the unit sphere."""
    lat = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitudes, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


@functools.lru_cache(maxsize=4)
def _load_fields(request: str) -> tuple[Any, dict[str, np.ndarray]]:
    """Retrieve the fields of a MARS request and index their grid.

    The result is cached, so the fields are only retrieved once per process,
    even though the filter is re-created for every group of dates.

    Parameters
    ----------
    request : str
        The MARS request, as a JSON string (so it can be used as a cache key).

    Returns
    -------
    tuple
        A KD-tree of the grid points and a dictionary of values per parameter.
    """
    import earthkit.data as ekd
    from scipy.spatial import cKDTree

    fields = ekd.from_source("mars", json.loads(request))

    latitudes = longitudes = None
    values = {}
    for field in fields:
        param = field.metadata("param")
        if param in values:
            # The fields are expected to be constant in time, keep the first one
            continue

        lat, lon = field.grid_points()
        if latitudes is None:
            latitudes, longitudes = lat, lon
        elif lat.shape != latitudes.shape or not (np.allclose(lat, latitudes) and np.allclose(lon, longitudes)):
            raise ValueError(f"add_gridded_fields: field '{param}' is not on the same grid as the other fields")

        values[param] = field.to_numpy(flatten=True).astype(np.float64)

    if not values:
        raise ValueError(f"add_gridded_fields: no fields returned by MARS for {request}")

    LOG.info("add_gridded_fields: retrieved %s on %d grid points", sorted(values), latitudes.size)
    return cKDTree(_xyz(latitudes, longitudes)), values


@filter_registry.register("add_gridded_fields")
class AddGriddedFields(Filter):
    """Adds columns with the values of gridded fields at the observation locations.

    The fields are retrieved from MARS and sampled at the grid point nearest to
    each observation (great-circle distance). They are assumed to be constant in
    time (e.g. land-sea mask, orography): if the request returns several fields
    for a parameter, only the first one is used. The new columns are named after
    the MARS ``param`` short names.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - add_gridded_fields:
              mars:
                class: ea
                expver: "0001"
                stream: oper
                type: an
                levtype: sfc
                param: [lsm, z, sdor, slor]
                grid: n320
                date: 2023-12-01
                time: "00:00"

    """

    def __init__(self, *, mars: dict, param: list[str] | None = None):
        if not mars:
            raise ValueError("add_gridded_fields: a MARS request must be given with the 'mars' key.")
        self.request = json.dumps(mars, sort_keys=True, default=str)
        self.param = param

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        for column in ("latitude", "longitude"):
            if column not in obs_df.columns:
                raise ValueError(f"add_gridded_fields: column '{column}' missing from {list(obs_df.columns)}")

        tree, values = _load_fields(self.request)

        params = self.param if self.param is not None else list(values)
        missing = [p for p in params if p not in values]
        if missing:
            raise ValueError(f"add_gridded_fields: {missing} not returned by MARS (got {sorted(values)})")

        existing = [p for p in params if p in obs_df.columns]
        if existing:
            raise ValueError(f"add_gridded_fields: columns {existing} already exist")

        obs_df = obs_df.copy()
        if len(obs_df) == 0:
            for p in params:
                obs_df[p] = pd.Series(dtype=np.float64)
            return obs_df

        _, index = tree.query(_xyz(obs_df["latitude"].values, obs_df["longitude"].values))
        for p in params:
            obs_df[p] = values[p][index]

        return obs_df
