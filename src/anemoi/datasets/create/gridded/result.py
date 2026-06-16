# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
import math
import time
from collections import defaultdict
from functools import cached_property
from typing import Any
from typing import DefaultDict

import numpy as np
from anemoi.utils.dates import as_timedelta
from anemoi.utils.humanize import seconds_to_human
from anemoi.utils.humanize import shorten_list
from earthkit.data.core.order import build_remapping

from anemoi.datasets.create.input.result import Result

LOG = logging.getLogger(__name__)
QUIET = set()


# Synthetic variable names with known metadata flags.
# These are computed forcings that do not come from GRIB fields.
_KNOWN_VARIABLES: dict[str, dict[str, bool]] = {
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


def _fields_metatata(variables: tuple[str, ...], cube: Any, units_seen: dict) -> dict[str, Any]:
    """Retrieve metadata for the given variables and cube.

    Parameters
    ----------
    variables : tuple of str
        The variables to retrieve metadata for.
    cube : Any
        The data cube.
    units_seen : dict
        Mapping of variable name to its first-seen units, used to enforce that
        all fields of a given variable share the same units.

    Returns
    -------
    dict
        The metadata dictionary.
    """
    assert isinstance(variables, tuple), variables

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

    # Find the axis that corresponds to the variables dimension — the one whose
    # values are the variable names.  This is position-agnostic so it works for
    # both 3-key (gridded) and 5-key (trajectories, [date, time, step,
    # param_level, number]) order_by configurations.
    variables_set = set(variables)
    variables_axis_idx: int | None = None
    for axis_idx, axis_name in enumerate(cube.user_coords.keys()):
        axis_values = cube.user_coords[axis_name]
        if axis_values and axis_values[0] in variables_set:
            variables_axis_idx = axis_idx
            break
    if variables_axis_idx is None:
        raise ValueError(
            f"Could not find variables axis among {list(cube.user_coords.keys())}; " f"variables={variables}"
        )

    # We hold the first axis fixed (so we only walk one "row" of the cube) and
    # collect one representative field per variable.
    seen_variables: set[str] = set()
    primary_axis_value: Any = None
    for c in cube.iterate_cubelets():

        if primary_axis_value is None:
            primary_axis_value = c._coords_names[0]

        if primary_axis_value != c._coords_names[0]:
            continue

        current_variable = c._coords_names[variables_axis_idx]

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
            # in such case of incorrect encoding, P1 refers to endStep and P2 to startStep.
            # Note that this is, on purpose, the opposite of the usual convention.
            endStep = as_timedelta(f.metadata("P1"))
            startStep = as_timedelta(f.metadata("P2"))

        if startStep is not None and endStep is not None:
            assert endStep >= startStep, (startStep, endStep, md)

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
                    f" {stepTypeForConversion=} ({STEP_TYPE_FOR_CONVERSION.get(stepTypeForConversion)}),"
                    f" {typeOfStatisticalProcessing=} ({TYPE_OF_STATISTICAL_PROCESSING.get(typeOfStatisticalProcessing)}),"
                    f" {timeRangeIndicator=} ({TIME_RANGE_INDICATOR.get(timeRangeIndicator)})"
                )

            # print(md["param"], "startStep", startStep, "endStep", endStep, "process", process, "typeOfStatisticalProcessing", typeOfStatisticalProcessing)
            other[current_variable]["process"] = process
            other[current_variable]["period"] = (startStep, endStep)

        units = f.metadata("units", default=None)
        if current_variable in units_seen:
            if units_seen[current_variable] != units:
                raise ValueError(
                    f"Variable {current_variable} has multiple units: {units_seen[current_variable]} and {units}"
                )
        else:
            units_seen[current_variable] = units

        if units is None and current_variable not in QUIET:
            LOG.warning(f"Cannot establish units for variable '{current_variable}'.")
            QUIET.add(current_variable)

        other[current_variable]["units"] = units

        grib = {k: f.metadata(k, default=None) for k in ("paramId", "shortName")}
        if any(grib.values()):
            other[current_variable]["grib"] = grib

        for k in md.copy().keys():
            if k.startswith("_"):
                md.pop(k)

        if current_variable in mars:
            mars[current_variable] = _merge(md, mars[current_variable])
        else:
            mars[current_variable] = md

        seen_variables.add(current_variable)

    result: dict[str, dict[str, Any]] = {}
    for k, v in mars.items():
        result[k] = dict(mars=v) if v else {}
        result[k].update(other[k])
        result[k].update(_KNOWN_VARIABLES.get(k, {}))
        # assert result[k], k

    assert seen_variables == variables_set, (seen_variables, variables_set)
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


class GriddedResult(Result):
    """Shared implementation for simple and trajectory gridded result classes."""

    empty: bool = False
    _coords_already_built: bool = False

    def __init__(self, context: Any, argument: Any, datasource: Any) -> None:
        """Initialise a GriddedResult instance.

        Parameters
        ----------
        context : Any
            The context in which the result is created.
        argument : Any
            The group of dates for the result.
        datasource : Any
            The data source for the result.
        """

        from anemoi.datasets.create.arguments import ForecastDates
        from anemoi.datasets.dates.groups import GroupOfDates

        self.context: Any = context
        self.datasource = datasource
        self.group_of_dates = argument
        assert isinstance(
            self.group_of_dates, (GroupOfDates, ForecastDates)
        ), f"Expected group_of_dates to be a GroupOfDates or ForecastDates, got {type(self.group_of_dates)}: {self.group_of_dates}"

        self._origins = []
        # Used to check if units are consistent across fields for the same variable
        self._past_units = {}

    @property
    def data_request(self) -> dict[str, Any]:
        """Return a dictionary with the parameters needed to retrieve the data."""
        return _data_request(self.datasource)

    @property
    def origins(self) -> dict[str, Any]:
        """Return a dictionary with the parameters needed to retrieve the data origins."""
        return {"version": 1, "origins": self._origins}

    def get_cube(self) -> Any:
        """Retrieve the data cube for the result.

        Returns
        -------
        Any
            The data cube.
        """

        ds: Any = self.datasource

        self.remapping: Any = self.context.remapping
        self.order_by: Any = self.context.order_by
        self.start: float = time.time()
        LOG.info("Sorting dataset %s %s", self.order_by, self.remapping)
        assert self.order_by, self.order_by

        self.patches: dict[str, dict[Any | None, int]] = {"number": {None: 0}}

        try:
            cube: Any = ds.cube(
                self.order_by,
                remapping=self.remapping,
                flatten_values=True,
                patches=self.patches,
            )
            cube = cube.squeeze()
            LOG.debug(f"Sorting done in {seconds_to_human(time.time()-self.start)}.")
        except AttributeError:
            import pandas as pd

            if isinstance(ds, pd.DataFrame):
                raise ValueError(
                    "Did you meant to build a tabular dataset? Did you forget to specify 'layout: tabular' in your recipe?"
                )
            raise
        except ValueError:
            self.explain(ds, self.order_by, remapping=self.remapping, patches=self.patches)
            raise

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("Cube shape: %s", cube)
            for k, v in cube.user_coords.items():
                LOG.debug("  %s %s", k, shorten_list(v, max_length=10))

        return cube

    def explain(self, ds: Any, *args: Any, remapping: Any, patches: Any) -> None:
        """Explain the data cube creation process.

        Parameters
        ----------
        ds : Any
            The data source.
        args : Any
            Additional arguments.
        remapping : Any
            The remapping configuration.
        patches : Any
            The patches configuration.
        """
        METADATA: tuple[str, ...] = (
            "date",
            "time",
            "step",
            "hdate",
            "valid_datetime",
            "levtype",
            "levelist",
            "number",
            "level",
            "shortName",
            "paramId",
            "variable",
        )

        # We redo the logic here
        print()
        print("❌" * 40)
        print()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]

        # print("Executing", self.action_path)
        # print("Dates:", compress_dates(self.dates))

        names: list[str] = []
        for a in args:
            if isinstance(a, str):
                names.append(a)
            elif isinstance(a, dict):
                names += list(a.keys())

        print(f"Building a {len(names)}D hypercube using", names)
        ds = ds.order_by(*args, remapping=remapping, patches=patches)
        user_coords = ds.unique_values(*names, remapping=remapping, patches=patches, progress_bar=False)

        print()
        print("Number of unique values found for each coordinate:")
        for k, v in user_coords.items():
            print(f"  {k:20}:", len(v))
            for n in sorted(v):
                print("     ", n)

        print()
        user_shape: tuple[int, ...] = tuple(len(v) for k, v in user_coords.items())
        print("Shape of the hypercube           :", user_shape)
        print(
            "Number of expected fields        :", math.prod(user_shape), "=", " x ".join([str(i) for i in user_shape])
        )
        print("Number of fields in the dataset  :", len(ds))
        print("Difference                       :", abs(len(ds) - math.prod(user_shape)))
        print()

        remapping = build_remapping(remapping, patches)
        expected = set(itertools.product(*user_coords.values()))
        extra = set()

        if math.prod(user_shape) > len(ds):
            print(f"This means that all the fields in the datasets do not exists for all combinations of {names}.")

            for f in ds:
                metadata = remapping(f.metadata)
                key = tuple(metadata(n, default=None) for n in names)
                if key in expected:
                    expected.remove(key)
                else:
                    extra.add(key)

            print("Missing fields:")
            print()
            for i, f in enumerate(sorted(expected)):
                print(" ", f)
                if i >= 9 and len(expected) > 10:
                    print("...", len(expected) - i - 1, "more")
                    break

            print("Extra fields:")
            print()
            for i, f in enumerate(sorted(extra)):
                print(" ", f)
                if i >= 9 and len(extra) > 10:
                    print("...", len(extra) - i - 1, "more")
                    break

            print()
            print("Missing values:")
            per_name = defaultdict(set)
            for e in expected:
                for n, v in zip(names, e):
                    per_name[n].add(v)

            for n, v in per_name.items():
                print(" ", n, len(v), shorten_list(sorted(v), max_length=10))
            print()

            print("Extra values:")
            per_name = defaultdict(set)
            for e in extra:
                for n, v in zip(names, e):
                    per_name[n].add(v)

            for n, v in per_name.items():
                print(" ", n, len(v), shorten_list(sorted(v), max_length=10))
            print()

            print("To solve this issue, you can:")
            print(
                "  - Provide a better selection, like 'step: 0' or 'level: 1000' to "
                "reduce the number of selected fields."
            )
            print(
                "  - Split the 'input' part in smaller sections using 'join', "
                "making sure that each section represent a full hypercube."
            )

        else:
            print(f"More fields in dataset that expected for {names}. " "This means that some fields are duplicated.")
            duplicated = defaultdict(list)
            for f in ds:
                # print(f.metadata(namespace="default"))
                metadata = remapping(f.metadata)
                key = tuple(metadata(n, default=None) for n in names)
                duplicated[key].append(f)

            print("Duplicated fields:")
            print()
            duplicated = {k: v for k, v in duplicated.items() if len(v) > 1}
            for i, (k, v) in enumerate(sorted(duplicated.items())):
                print(" ", k)
                for f in v:
                    x = {k: f.metadata(k, default=None) for k in METADATA if f.metadata(k, default=None) is not None}
                    print("   ", f, x)
                if i >= 9 and len(duplicated) > 10:
                    print("...", len(duplicated) - i - 1, "more")
                    break

            print()
            print("To solve this issue, you can:")
            print("  - Provide a better selection, like 'step: 0' or 'level: 1000'")
            print("  - Change the way 'param' is computed using 'variable_naming' " "in the 'build' section.")

        print()
        print("❌" * 40)
        print()

    def _post_build_coords(self, from_data: Any, keys: list[str]) -> None:
        """Hook for subclasses to extract extra coordinates from the cube.

        Called at the end of ``build_coords()`` before the guard flag is set.
        Override to capture additional coordinate arrays (e.g. step values for
        trajectory datasets).

        Parameters
        ----------
        from_data : Any
            The ``cube.user_coords`` mapping.
        keys : list of str
            The ordered coordinate key names from ``context.order_by``.
        """

    def build_coords(self) -> None:
        """Build the coordinate arrays for the result if not already built."""
        if self._coords_already_built:
            return

        cube: Any = self.get_cube()

        from_data: Any = cube.user_coords
        from_config: Any = {k: "ascending" for k in self.context.order_by}

        keys_from_config: list = list(from_config.keys())
        keys_from_data: list = list(from_data.keys())
        assert keys_from_data == keys_from_config, f"Critical error: {keys_from_data=} != {keys_from_config=}. {self=}"

        # The last two order_by keys are always (variables, ensembles);
        # extra leading keys (e.g. 'step' for trajectories) sit before them.
        variables_key: str = keys_from_config[-2]
        ensembles_key: str = keys_from_config[-1]

        if isinstance(from_config[variables_key], (list, tuple)):
            assert all([v == w for v, w in zip(from_data[variables_key], from_config[variables_key])]), (
                from_data[variables_key],
                from_config[variables_key],
            )

        self._variables: Any = from_data[variables_key]  # "param_level"
        self._ensembles: Any = from_data[ensembles_key]  # "number"

        self._post_build_coords(from_data, keys_from_config)

        first_field: Any = self.datasource[0]
        grid_points: Any = first_field.grid_points()

        lats: Any = grid_points[0]
        lons: Any = grid_points[1]

        assert len(lats) == len(lons), (len(lats), len(lons), first_field)
        assert len(lats) == math.prod(first_field.shape), (len(lats), first_field.shape, first_field)

        north: float = np.amax(lats)
        south: float = np.amin(lats)
        east: float = np.amax(lons)
        west: float = np.amin(lons)

        assert -90 <= south <= north <= 90, (south, north, first_field)
        assert (-180 <= west <= east <= 180) or (0 <= west <= east <= 360), (
            west,
            east,
            first_field,
        )

        grid_values: list = list(range(len(grid_points[0])))

        self._grid_points: Any = grid_points
        self._resolution: Any = first_field.resolution
        if self._resolution is None:
            try:
                self._resolution = first_field.metadata().get("resolution")
            except Exception:
                pass
        self._grid_values: Any = grid_values
        self._field_shape: Any = first_field.shape
        self._proj_string: Any = first_field.proj_string if hasattr(first_field, "proj_string") else None

        self._cube: Any = cube

        self._coords_already_built: bool = True

    @property
    def variables(self) -> list[str]:
        """Retrieve the variables for the result."""
        self.build_coords()
        return self._variables

    @property
    def variables_metadata(self) -> dict[str, Any]:
        """Retrieve the metadata for the variables."""
        return _fields_metatata(self.variables, self._cube, self._past_units)

    @property
    def typed_variables(self) -> dict[str, Any]:
        """Retrieve the typed metadata for the variables."""
        from anemoi.transform.variables import Variable

        return {k: Variable.from_dict(k, v) for k, v in self.variables_metadata.items()}

    @property
    def ensembles(self) -> Any:
        """Retrieve the ensembles for the result."""
        self.build_coords()
        return self._ensembles

    @property
    def resolution(self) -> Any:
        """Retrieve the resolution for the result."""
        self.build_coords()
        return self._resolution

    @property
    def grid_values(self) -> Any:
        """Retrieve the grid values for the result."""
        self.build_coords()
        return self._grid_values

    @property
    def grid_points(self) -> Any:
        """Retrieve the grid points for the result."""
        self.build_coords()
        return self._grid_points

    @property
    def field_shape(self) -> Any:
        """Retrieve the field shape for the result."""
        self.build_coords()
        return self._field_shape

    @property
    def proj_string(self) -> Any:
        """Retrieve the projection string for the result."""
        self.build_coords()
        return self._proj_string

    @cached_property
    def shape(self) -> list[int]:
        """Retrieve the shape of the result."""
        return (
            len(self.group_of_dates),
            len(self.variables),
            len(self.ensembles),
            len(self.grid_values),
        )


class SimpleGriddedResult(GriddedResult):
    """Result class for simple (valid-date-indexed) gridded datasets."""

    @cached_property
    def coords(self) -> dict[str, Any]:
        """Retrieve the coordinates of the result."""
        return {
            "dates": list(self.group_of_dates),
            "variables": self.variables,
            "ensembles": self.ensembles,
            "values": self.grid_values,
        }
