# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from collections import defaultdict

from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo import thermo

from .single_level_specific_humidity_to_relative_humidity import NewDataField


def execute(context, input, t, rh, q="q"):
    """Convert relative humidity on pressure levels to specific humidity"""
    result = FieldArray()

    params = (t, rh)
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
        rh_pl = values[rh].to_numpy(flatten=True)
        pressure = keys[4][1] * 100  # TODO: REMOVE HARDCODED INDICES
        # print(f"Handling fields for pressure level {pressure}...")

        # actual conversion from rh --> q_v
        q_pl = thermo.specific_humidity_from_relative_humidity(t_pl, rh_pl, pressure)
        result.append(NewDataField(values[rh], q_pl, q))

    return result
