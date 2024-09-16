# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from collections import defaultdict

import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo.wind.array import xy_to_polar

from anemoi.datasets.create.functions.filters.speeddir_to_uv import NewDataField


def execute(context, input, u_component, v_component, wind_speed, wind_dir, in_radians=False):
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
