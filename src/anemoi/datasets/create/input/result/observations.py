# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections import defaultdict
from typing import Any
from typing import DefaultDict

from anemoi.utils.dates import as_timedelta

from . import Result

LOG = logging.getLogger(__name__)


def _fields_metatata(variables: tuple[str, ...], cube: Any) -> dict[str, Any]:
    """Retrieve metadata for the given variables and cube.

    Parameters
    ----------
    variables : tuple of str
        The variables to retrieve metadata for.
    cube : Any
        The data cube.

    Returns
    -------
    dict
        The metadata dictionary.
    """
    assert isinstance(variables, tuple), variables

    KNOWN: dict[str, dict[str, bool]] = {
        "cos_julian_day": dict(computed_forcing=True, constant_in_time=False),
        "cos_latitude": dict(computed_forcing=True, constant_in_time=True),
        "cos_local_time": dict(computed_forcing=True, constant_in_time=False),
        "cos_longitude": dict(computed_forcing=True, constant_in_time=True),
        "cos_solar_zenith_angle": dict(computed_forcing=True, constant_in_time=False),
        "insolation": dict(computed_forcing=True, constant_in_time=False),
        "latitude": dict(computed_forcing=True, constant_in_time=True),
        "longitude": dict(computed_forcing=True, constant_in_time=True),
        "sin_julian_day": dict(computed_forcing=True, constant_in_time=False),
        "sin_latitude": dict(computed_forcing=True, constant_in_time=True),
        "sin_local_time": dict(computed_forcing=True, constant_in_time=False),
        "sin_longitude": dict(computed_forcing=True, constant_in_time=True),
    }

    def _merge(md1: dict[str, Any], md2: dict[str, Any]) -> dict[str, Any]:
        assert set(md1.keys()) == set(md2.keys()), (set(md1.keys()), set(md2.keys()))
        result: dict[str, Any] = {}
        for k in md1.keys():
            v1 = md1[k]
            v2 = md2[k]

            if v1 == v2:
                result[k] = v1
                continue

            if isinstance(v1, list):
                assert v2 not in v1, (v1, v2)
                result[k] = sorted(v1 + [v2])
                continue

            if isinstance(v2, list):
                assert v1 not in v2, (v1, v2)
                result[k] = sorted(v2 + [v1])
                continue

            result[k] = sorted([v1, v2])

        return result

    mars: dict[str, Any] = {}
    other: DefaultDict[str, dict[str, Any]] = defaultdict(dict)
    i: int = -1
    date: str | None = None
    for c in cube.iterate_cubelets():

        if date is None:
            date = c._coords_names[0]

        if date != c._coords_names[0]:
            continue

        if i == -1 or c._coords_names[1] != variables[i]:
            i += 1

        f = cube[c.coords]
        md = f.metadata(namespace="mars")
        if not md:
            md = f.metadata(namespace="default")

        if md.get("param") == "~":
            md["param"] = f.metadata("param")
            assert md["param"] not in ("~", "unknown"), (md, f.metadata("param"))

        if md.get("param") == "unknown":
            md["param"] = str(f.metadata("paramId", default="unknown"))
            # assert md['param'] != 'unknown', (md, f.metadata('param'))

        startStep = f.metadata("startStep", default=None)
        if startStep is not None:
            startStep = as_timedelta(startStep)

        endStep = f.metadata("endStep", default=None)
        if endStep is not None:
            endStep = as_timedelta(endStep)

        stepTypeForConversion = f.metadata("stepTypeForConversion", default=None)
        typeOfStatisticalProcessing = f.metadata("typeOfStatisticalProcessing", default=None)
        timeRangeIndicator = f.metadata("timeRangeIndicator", default=None)

        # GRIB1 precipitation accumulations are not correctly encoded
        if startStep == endStep and stepTypeForConversion == "accum":
            endStep = f.metadata("P1")
            startStep = f.metadata("P2")

        if startStep != endStep:
            # https://codes.ecmwf.int/grib/format/grib2/ctables/4/10/
            TYPE_OF_STATISTICAL_PROCESSING: dict[int | None, str | None] = {
                None: None,
                0: "average",
                1: "accumulation",
                2: "maximum",
                3: "minimum",
                4: "difference(end-start)",
                5: "root_mean_square",
                6: "standard_deviation",
                7: "covariance",
                8: "difference(start-end)",
                9: "ratio",
                10: "standardized_anomaly",
                11: "summation",
                100: "severity",
                101: "mode",
            }

            # https://codes.ecmwf.int/grib/format/grib1/ctable/5/

            TIME_RANGE_INDICATOR: dict[int, str] = {
                4: "accumulation",
                3: "average",
            }

            STEP_TYPE_FOR_CONVERSION: dict[str, str] = {
                "min": "minimum",
                "max": "maximum",
                "accum": "accumulation",
            }

            #
            # A few patches
            #

            PATCHES: dict[str, str] = {
                "10fg6": "maximum",
                "mntpr3": "minimum",  # Not in param db
                "mntpr6": "minimum",  # Not in param db
                "mxtpr3": "maximum",  # Not in param db
                "mxtpr6": "maximum",  # Not in param db
            }

            process = TYPE_OF_STATISTICAL_PROCESSING.get(typeOfStatisticalProcessing)
            if process is None:
                process = TIME_RANGE_INDICATOR.get(timeRangeIndicator)
            if process is None:
                process = STEP_TYPE_FOR_CONVERSION.get(stepTypeForConversion)
            if process is None:
                process = PATCHES.get(md["param"])
                if process is not None:
                    LOG.error(f"Unknown process {stepTypeForConversion} for {md['param']}, using {process} instead")

            if process is None:
                raise ValueError(
                    f"Unknown for {md['param']}:"
                    f" {stepTypeForConversion=} ({STEP_TYPE_FOR_CONVERSION.get('stepTypeForConversion')}),"
                    f" {typeOfStatisticalProcessing=} ({TYPE_OF_STATISTICAL_PROCESSING.get(typeOfStatisticalProcessing)}),"
                    f" {timeRangeIndicator=} ({TIME_RANGE_INDICATOR.get(timeRangeIndicator)})"
                )

            # print(md["param"], "startStep", startStep, "endStep", endStep, "process", process, "typeOfStatisticalProcessing", typeOfStatisticalProcessing)
            other[variables[i]]["process"] = process
            other[variables[i]]["period"] = (startStep, endStep)

        for k in md.copy().keys():
            if k.startswith("_"):
                md.pop(k)

        if variables[i] in mars:
            mars[variables[i]] = _merge(md, mars[variables[i]])
        else:
            mars[variables[i]] = md

    result: dict[str, dict[str, Any]] = {}
    for k, v in mars.items():
        result[k] = dict(mars=v) if v else {}
        result[k].update(other[k])
        result[k].update(KNOWN.get(k, {}))
        # assert result[k], k

    assert i + 1 == len(variables), (i + 1, len(variables))
    return result


def _data_request(data: Any) -> dict[str, Any]:
    """Build a data request dictionary from the given data.

    Parameters
    ----------
    data : Any
        The data to build the request from.

    Returns
    -------
    dict
        The data request dictionary.
    """
    date: Any | None = None
    params_levels: DefaultDict[str, set] = defaultdict(set)
    params_steps: DefaultDict[str, set] = defaultdict(set)

    area: Any | None = None
    grid: Any | None = None

    for field in data:
        try:
            if date is None:
                date = field.metadata("valid_datetime")

            if field.metadata("valid_datetime") != date:
                continue

            as_mars = field.metadata(namespace="mars")
            if not as_mars:
                continue
            step = as_mars.get("step")
            levtype = as_mars.get("levtype", "sfc")
            param = as_mars["param"]
            levelist = as_mars.get("levelist", None)
            area = field.mars_area
            grid = field.mars_grid

            if levelist is None:
                params_levels[levtype].add(param)
            else:
                params_levels[levtype].add((param, levelist))

            if step:
                params_steps[levtype].add((param, step))
        except Exception:
            LOG.error(f"Error in retrieving metadata (cannot build data request info) for {field}", exc_info=True)

    def sort(old_dic: DefaultDict[str, set]) -> dict[str, list[Any]]:
        new_dic: dict[str, list[Any]] = {}
        for k, v in old_dic.items():
            new_dic[k] = sorted(list(v))
        return new_dic

    params_steps = sort(params_steps)
    params_levels = sort(params_levels)

    return dict(param_level=params_levels, param_step=params_steps, area=area, grid=grid)


class ObservationsResult(Result):

    def __init__(self, context: Any, datasource: Any) -> None:

        pass
