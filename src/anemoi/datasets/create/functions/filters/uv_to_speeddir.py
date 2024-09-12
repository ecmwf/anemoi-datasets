# Written by Michiel Van Ginderachter (mpvginde)
# Date: 2024-06-14

from collections import defaultdict

import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo.wind.array import xy_to_polar


class NewDataField:
    def __init__(self, field, data, new_name):
        self.field = field
        self.data = data
        self.new_name = new_name

    def to_numpy(self, *args, **kwargs):
        return self.data

    def metadata(self, key):
        value = self.field.metadata(key)
        if key == "param":
            return self.new_name
        return value

    def __getattr__(self, name):
        return getattr(self.field, name)


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
        magnitude, direction = xy_to_polar(u.to_numpy(reshape=False), v.to_numpy(reshape=False))
        if in_radians:
            direction = np.deg2rad(direction)

        result.append(NewDataField(u, magnitude, wind_speed))
        result.append(NewDataField(v, direction, wind_dir))

    return result
