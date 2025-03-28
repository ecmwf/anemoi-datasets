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

import earthkit.data as ekd
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list

from .legacy import legacy_filter


@legacy_filter(__file__)
def execute(context: Any, input: ekd.FieldList, wz: str, t: str, w: str = "w") -> ekd.FieldList:
    """Convert geometric vertical velocity (m/s) to vertical velocity (Pa / s).

    Parameters
    ----------
    context : Any
        The context for the execution.
    input : List[Any]
        The list of input fields.
    wz : str
        The parameter name for geometric vertical velocity.
    t : str
        The parameter name for temperature.
    w : str, optional
        The parameter name for vertical velocity. Defaults to "w".

    Returns
    -------
    ekd.FieldList
        The resulting FieldArray with converted vertical velocity fields.
    """
    result = []

    params = (wz, t)
    pairs = defaultdict(dict)

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
        else:
            result.append(f)

    for keys, values in pairs.items():

        if len(values) != 2:
            raise ValueError("Missing fields")

        wz_pl = values[wz].to_numpy(flatten=True)
        t_pl = values[t].to_numpy(flatten=True)
        pressure = next(
            float(v) * 100 for k, v in keys if k in ["level", "levelist"]
        )  # Looks first for "level" then "levelist" value
        w_pl = wz_to_w(wz_pl, t_pl, pressure)
        result.append(new_field_from_numpy(values[wz], w_pl, param=w))

    return new_fieldlist_from_list(result)


def wz_to_w(wz: Any, t: Any, pressure: float) -> Any:
    """Convert geometric vertical velocity (m/s) to vertical velocity (Pa / s).

    Parameters
    ----------
    wz : Any
        The geometric vertical velocity data.
    t : Any
        The temperature data.
    pressure : float
        The pressure value.

    Returns
    -------
    Any
        The vertical velocity data in Pa / s.
    """
    g = 9.81
    Rd = 287.058

    return -wz * g * pressure / (t * Rd)
