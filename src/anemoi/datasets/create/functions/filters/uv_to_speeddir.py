# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""This module provides a function to convert u and v wind components to wind speed and direction."""

from collections import defaultdict
from typing import Any
from typing import List

import earthkit.data as ekd
import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo.wind.array import xy_to_polar

from anemoi.datasets.create.functions.filters.speeddir_to_uv import NewDataField


def execute(
    context: Any,
    input: List[Any],
    u_component: str,
    v_component: str,
    wind_speed: str,
    wind_dir: str,
    in_radians: bool = False,
) -> ekd.FieldList:
    """Converts u and v wind components to wind speed and direction.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : List[Any]
        List of input fields containing wind components.
    u_component : str
        The name of the u component field.
    v_component : str
        The name of the v component field.
    wind_speed : str
        The name of the wind speed field to be created.
    wind_dir : str
        The name of the wind direction field to be created.
    in_radians : bool, optional
        If True, the wind direction is returned in radians. Default is False.

    Returns
    -------
    ekd.FieldList
        A FieldArray containing the wind speed and direction fields.
    """
    result = FieldArray()

    wind_params = (u_component, v_component)
    wind_pairs = defaultdict(dict)

    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")

        if param not in wind_params:
            result.append(f)
            continue

        key = tuple(key.items())

        if param in wind_pairs[key]:
            raise ValueError(f"Duplicate wind component {param} for {key}")

        wind_pairs[key][param] = f

    for _, pairs in wind_pairs.items():
        if len(pairs) != 2:
            raise ValueError("Missing wind component")

        u = pairs[u_component]
        v = pairs[v_component]

        # assert speed.grid_mapping == dir.grid_mapping
        magnitude, direction = xy_to_polar(u.to_numpy(flatten=True), v.to_numpy(flatten=True))
        if in_radians:
            direction = np.deg2rad(direction)

        result.append(NewDataField(u, magnitude, wind_speed))
        result.append(NewDataField(v, direction, wind_dir))

    return result
