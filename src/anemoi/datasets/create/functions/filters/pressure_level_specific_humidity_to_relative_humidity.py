# Written by Michiel Van Ginderachter (mpvginde)
# Date: 2024-06-27

from collections import defaultdict

import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo import thermo

from anemoi.datasets.create.functions.filters.single_level_specific_humidity_to_relative_humidity import NewDataField


def execute(context, input, t, q, rh="r"):
    """Convert specific humidity on pressure levels to relative humidity"""
    result = FieldArray()

    params = (t, q)
    pairs = defaultdict(dict)

    # Gather all necessary fields
    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        if param in params:
            key = tuple(key.items())

            if param in pairs[key]:
                raise ValueError(f"Duplicate field {param} for {key}")

            pairs[key][param] = f
            if param == t:
                result.append(f)
        # all other parameters
        else:
            result.append(f)

    for keys, values in pairs.items():
        # some checks

        if len(values) != 2:
            raise ValueError("Missing fields")

        t_pl = values[t].to_numpy(flatten=True)
        q_pl = values[q].to_numpy(flatten=True)
        pressure = keys[4][1] * 100  # TODO: REMOVE HARDCODED INDICES
        # print(f"Handling fields for pressure level {pressure}...")

        # actual conversion from rh --> q_v
        rh_pl = thermo.relative_humidity_from_specific_humidity(t_pl, q_pl, pressure)
        result.append(NewDataField(values[q], rh_pl, rh))

    return result
