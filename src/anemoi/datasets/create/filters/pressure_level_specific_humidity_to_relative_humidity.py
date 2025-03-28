# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from typing import Any
from typing import Dict

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo import thermo

from .legacy import legacy_filter


@legacy_filter(__file__)
def execute(context: Any, input: ekd.FieldList, t: str, q: str, rh: str = "r") -> FieldArray:
    """Convert specific humidity on pressure levels to relative humidity.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : List[Any]
        List of input fields.
    t : str
        Temperature parameter.
    q : str
        Specific humidity parameter.
    rh : str, optional
        Relative humidity parameter. Defaults to "r".

    Returns
    -------
    ekd.FieldList
        Array of fields with relative humidity.
    """
    result = []
    params: tuple[str, str] = (t, q)
    pairs: Dict[tuple, Dict[str, Any]] = defaultdict(dict)

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
        pressure = next(
            float(v) * 100 for k, v in keys if k in ["level", "levelist"]
        )  # Looks first for "level" then "levelist" value
        # print(f"Handling fields for pressure level {pressure}...")

        # actual conversion from rh --> q_v
        rh_pl = thermo.relative_humidity_from_specific_humidity(t_pl, q_pl, pressure)
        result.append(new_field_from_numpy(values[q], rh_pl, param=rh))

    return new_fieldlist_from_list(result)
