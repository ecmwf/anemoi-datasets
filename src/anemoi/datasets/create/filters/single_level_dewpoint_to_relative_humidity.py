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
from earthkit.meteo import thermo

from .legacy import legacy_filter


@legacy_filter(__file__)
def execute(context: Any, input: ekd.FieldList, t: str, td: str, rh: str = "d") -> ekd.FieldList:
    """Convert dewpoint on single levels to relative humidity.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : List[Any]
        List of input fields.
    t : str
        Temperature parameter.
    td : str
        Dewpoint parameter.
    rh : str, optional
        Relative humidity parameter. Defaults to "d".

    Returns
    -------
    ekd.FieldList
        Array of fields with relative humidity.
    """
    result = []
    params: tuple[str, str] = (t, td)
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

        t_values = values[t].to_numpy(flatten=True)
        td_values = values[td].to_numpy(flatten=True)
        # actual conversion from td --> rh
        rh_values = thermo.relative_humidity_from_dewpoint(t=t_values, td=td_values)
        result.append(new_field_from_numpy(values[td], rh_values, param=rh))

    return new_fieldlist_from_list(result)
