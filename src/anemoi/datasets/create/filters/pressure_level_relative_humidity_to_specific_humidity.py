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
from typing import Tuple

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.meteo import thermo

from .legacy import legacy_filter


@legacy_filter(__file__)
def execute(context: Any, input: ekd.FieldList, t: str, rh: str, q: str = "q") -> ekd.FieldList:
    """Convert relative humidity on pressure levels to specific humidity.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : List[Any]
        List of input fields.
    t : str
        Temperature parameter.
    rh : str
        Relative humidity parameter.
    q : str, optional
        Specific humidity parameter. Defaults to "q".

    Returns
    -------
    ekd.FieldList
        Array of fields with specific humidity.
    """
    result = []
    params: Tuple[str, str] = (t, rh)
    pairs: Dict[Tuple[Any, ...], Dict[str, Any]] = defaultdict(dict)

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
        pressure = next(
            float(v) * 100 for k, v in keys if k in ["level", "levelist"]
        )  # Looks first for "level" then "levelist" value
        # print(f"Handling fields for pressure level {pressure}...")

        # actual conversion from rh --> q_v
        q_pl = thermo.specific_humidity_from_relative_humidity(t_pl, rh_pl, pressure)
        result.append(new_field_from_numpy(values[rh], q_pl, param=q))

    return new_fieldlist_from_list(result)
