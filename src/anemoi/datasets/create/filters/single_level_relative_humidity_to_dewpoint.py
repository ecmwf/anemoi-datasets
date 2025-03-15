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
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo import thermo

from .legacy import legacy_filter

EPS = 1.0e-4


@legacy_filter(__file__)
def execute(context: Any, input: ekd.FieldList, t: str, rh: str, td: str = "d") -> FieldArray:
    """Convert relative humidity on single levels to dewpoint.

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
    td : str, optional
        Dewpoint parameter. Defaults to "d".

    Returns
    -------
    FieldArray
        Array of fields with dewpoint.
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

        t_values = values[t].to_numpy(flatten=True)
        rh_values = values[rh].to_numpy(flatten=True)
        # Prevent 0 % Relative humidity which cannot be converted to dewpoint
        # Seems to happen over Egypt in the CERRA dataset
        rh_values[rh_values == 0] = EPS
        # actual conversion from rh --> td
        td_values = thermo.dewpoint_from_relative_humidity(t=t_values, r=rh_values)
        result.append(new_field_from_numpy(values[rh], td_values, param=td))

    return new_fieldlist_from_list(result)
