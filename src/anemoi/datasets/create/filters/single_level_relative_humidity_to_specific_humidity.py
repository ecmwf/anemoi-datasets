# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Dict
from typing import List

import earthkit.data as ekd
import numpy as np
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.meteo import thermo

from .legacy import legacy_filter
from .single_level_specific_humidity_to_relative_humidity import AutoDict
from .single_level_specific_humidity_to_relative_humidity import pressure_at_height_level


@legacy_filter(__file__)
def execute(
    context: Any,
    input: List[Any],
    height: float,
    t: str,
    rh: str,
    sp: str,
    new_name: str = "2q",
    **kwargs: Dict[str, Any],
) -> ekd.FieldList:
    """Convert the single (height) level relative humidity to specific humidity.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : List[Any]
        List of input fields.
    height : float
        The height level.
    t : str
        Temperature parameter name.
    rh : str
        Relative humidity parameter name.
    sp : str
        Surface pressure parameter name.
    new_name : str, optional
        The new name for the specific humidity field, by default "2q".
    **kwargs : Dict[str, Any]
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        The resulting field list with specific humidity fields.

    Raises
    ------
    KeyError
        If mandatory keys are missing.
    ValueError
        If there are duplicate fields or missing fields.
    """
    result = []

    MANDATORY_KEYS = ["A", "B"]
    OPTIONAL_KEYS = ["t_ml", "q_ml"]
    MISSING_KEYS = []
    DEFAULTS = dict(t_ml="t", q_ml="q")

    for key in OPTIONAL_KEYS:
        if key not in kwargs:
            print(f"key {key} not found in yaml-file, using default key: {DEFAULTS[key]}")
            kwargs[key] = DEFAULTS[key]

    for key in MANDATORY_KEYS:
        if key not in kwargs:
            MISSING_KEYS.append(key)

    if MISSING_KEYS:
        raise KeyError(f"Following keys are missing: {', '.join(MISSING_KEYS)}")

    single_level_params = (t, rh, sp)
    model_level_params = (kwargs["t_ml"], kwargs["q_ml"])

    needed_fields = AutoDict()

    # Gather all necessary fields
    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        # check single level parameters
        if param in single_level_params:
            levtype = key.pop("levtype")
            key = tuple(key.items())

            if param in needed_fields[key][levtype]:
                raise ValueError(f"Duplicate single level field {param} for {key}")

            needed_fields[key][levtype][param] = f
            if param == rh:
                if kwargs.get("keep_rh", False):
                    result.append(f)
            else:
                result.append(f)

        # check model level parameters
        elif param in model_level_params:
            levtype = key.pop("levtype")
            levelist = key.pop("levelist")
            key = tuple(key.items())

            if param in needed_fields[key][levtype][levelist]:
                raise ValueError(f"Duplicate model level field {param} for {key} at level {levelist}")

            needed_fields[key][levtype][levelist][param] = f

        # all other parameters
        else:
            result.append(f)

    for _, values in needed_fields.items():
        # some checks
        if len(values["sfc"]) != 3:
            raise ValueError("Missing surface fields")

        rh_sl = values["sfc"][rh].to_numpy(flatten=True)
        t_sl = values["sfc"][t].to_numpy(flatten=True)
        sp_sl = values["sfc"][sp].to_numpy(flatten=True)

        nlevels = len(kwargs["A"]) - 1
        if len(values["ml"]) != nlevels:
            raise ValueError("Missing model levels")

        for key in values["ml"].keys():
            if len(values["ml"][key]) != 2:
                raise ValueError(f"Missing field on level {key}")

        # create 3D arrays for upper air fields
        levels = list(values["ml"].keys())
        levels.sort()
        t_ml = []
        q_ml = []
        for level in levels:
            t_ml.append(values["ml"][level][kwargs["t_ml"]].to_numpy(flatten=True))
            q_ml.append(values["ml"][level][kwargs["q_ml"]].to_numpy(flatten=True))

        t_ml = np.stack(t_ml)
        q_ml = np.stack(q_ml)

        # actual conversion from rh --> q_v
        p_sl = pressure_at_height_level(height, q_ml, t_ml, sp_sl, np.array(kwargs["A"]), np.array(kwargs["B"]))
        q_sl = thermo.specific_humidity_from_relative_humidity(t_sl, rh_sl, p_sl)

        result.append(new_field_from_numpy(values["sfc"][rh], q_sl, param=new_name))

    return new_fieldlist_from_list(result)
