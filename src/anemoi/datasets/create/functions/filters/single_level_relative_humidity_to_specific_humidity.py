# Written by Michiel Van Ginderachter (mpvginde)
# Date: 2024-06-27


import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo import thermo
from single_level_specific_humidity_to_relative_humidity import AutoDict
from single_level_specific_humidity_to_relative_humidity import NewDataField
from single_level_specific_humidity_to_relative_humidity import pressure_at_height_level


def execute(context, input, height, t, rh, sp, new_name="2q", **kwargs):
    """Convert the single (height) level relative humidity to specific humidity"""
    result = FieldArray()

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

        result.append(NewDataField(values["sfc"][rh], q_sl, new_name))

    return result


if __name__ == "__main__":
    from earthkit.data import from_source
    from earthkit.data.readers.grib.index import GribFieldList

    # IFS forecast have both specific humidity and dewpoint
    sl = from_source(
        "mars",
        {
            "date": "2022-01-01",
            "class": "od",
            "expver": "1",
            "stream": "oper",
            "levtype": "sfc",
            "param": "96.174/134.128/167.128/168.128",
            "time": "00:00:00",
            "type": "fc",
            "step": "2",
            "grid": "O640",
        },
    )

    ml = from_source(
        "mars",
        {
            "date": "2022-01-01",
            "class": "od",
            "expver": "1",
            "stream": "oper",
            "levtype": "ml",
            "levelist": "130/131/132/133/134/135/136/137",
            "param": "130/133",
            "time": "00:00:00",
            "type": "fc",
            "step": "2",
            "grid": "O640",
        },
    )
    source = GribFieldList.merge([sl, ml])
