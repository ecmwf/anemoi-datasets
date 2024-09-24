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

EPS = 1.0e-4


def execute(context, input, t, rh, td="d"):
    """Convert relative humidity on single levels to dewpoint"""
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

        t_values = values[t].to_numpy(flatten=True)
        rh_values = values[rh].to_numpy(flatten=True)
        # Prevent 0 % Relative humidity which cannot be converted to dewpoint
        # Seems to happen over Egypt in the CERRA dataset
        rh_values[rh_values == 0] = EPS
        # actual conversion from rh --> td
        td_values = thermo.dewpoint_from_relative_humidity(t=t_values, r=rh_values)
        result.append(NewDataField(values[rh], td_values, td))

    return result
