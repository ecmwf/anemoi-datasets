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

import numpy as np
from anemoi.utils.humanize import seconds_to_human
from anemoi.utils.humanize import shorten_list
from earthkit.data.core.order import build_remapping

from .trace import trace
from .trace import trace_datasource

LOG = logging.getLogger(__name__)


def _fields_metatata(variables, cube):
    assert isinstance(variables, tuple), variables

    KNOWN = {
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

    def _merge(md1, md2):
        assert set(md1.keys()) == set(md2.keys()), (set(md1.keys()), set(md2.keys()))
        result = {}
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

    mars = {}
    other = defaultdict(dict)
    i = -1
    date = None
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
        assert startStep is None or isinstance(startStep, int), (startStep, type(f))

        endStep = f.metadata("endStep", default=None)
        assert endStep is None or isinstance(endStep, int), endStep

        stepTypeForConversion = f.metadata("stepTypeForConversion", default=None)
        typeOfStatisticalProcessing = f.metadata("typeOfStatisticalProcessing", default=None)
        timeRangeIndicator = f.metadata("timeRangeIndicator", default=None)

        # GRIB1 precipitation accumulations are not correctly encoded
        if startStep == endStep and stepTypeForConversion == "accum":
            startStep = 0

        if startStep != endStep:
            # https://codes.ecmwf.int/grib/format/grib2/ctables/4/10/
            TYPE_OF_STATISTICAL_PROCESSING = {
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

            TIME_RANGE_INDICATOR = {
                4: "accumulation",
                3: "average",
            }

            STEP_TYPE_FOR_CONVERSION = {
                "min": "minimum",
                "max": "maximum",
                "accum": "accumulation",
            }

            #
            # A few patches
            #

            PATCHES = {
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

    result = {}
    for k, v in mars.items():
        result[k] = dict(mars=v) if v else {}
        result[k].update(other[k])
        result[k].update(KNOWN.get(k, {}))
        assert result[k], k

    assert i + 1 == len(variables), (i + 1, len(variables))
    return result


def _data_request(data):
    date = None
    params_levels = defaultdict(set)
    params_steps = defaultdict(set)

    area = grid = None

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

    def sort(old_dic):
        new_dic = {}
        for k, v in old_dic.items():
            new_dic[k] = sorted(list(v))
        return new_dic

    params_steps = sort(params_steps)
    params_levels = sort(params_levels)

    return dict(param_level=params_levels, param_step=params_steps, area=area, grid=grid)


class Result:
    empty = False
    _coords_already_built = False

    def __init__(self, context, action_path, dates):
        from anemoi.datasets.dates.groups import GroupOfDates

        from .action import ActionContext

        assert isinstance(dates, GroupOfDates), dates

        assert isinstance(context, ActionContext), type(context)
        assert isinstance(action_path, list), action_path

        self.context = context
        self.group_of_dates = dates
        self.action_path = action_path

    @property
    @trace_datasource
    def datasource(self):
        self._raise_not_implemented()

    @property
    def data_request(self):
        """Returns a dictionary with the parameters needed to retrieve the data."""
        return _data_request(self.datasource)

    def get_cube(self):
        trace("🧊", f"getting cube from {self.__class__.__name__}")
        ds = self.datasource

        remapping = self.context.remapping
        order_by = self.context.order_by
        flatten_grid = self.context.flatten_grid
        start = time.time()
        LOG.debug("Sorting dataset %s %s", dict(order_by), remapping)
        assert order_by, order_by

        patches = {"number": {None: 0}}

        try:
            cube = ds.cube(
                order_by,
                remapping=remapping,
                flatten_values=flatten_grid,
                patches=patches,
            )
            cube = cube.squeeze()
            LOG.debug(f"Sorting done in {seconds_to_human(time.time()-start)}.")
        except ValueError:
            self.explain(ds, order_by, remapping=remapping, patches=patches)
            # raise ValueError(f"Error in {self}")
            exit(1)

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("Cube shape: %s", cube)
            for k, v in cube.user_coords.items():
                LOG.debug("  %s %s", k, shorten_list(v, max_length=10))

        return cube

    def explain(self, ds, *args, remapping, patches):

        METADATA = (
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

        names = []
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
            print(f"  {k:20}:", len(v), shorten_list(v, max_length=10))
        print()
        user_shape = tuple(len(v) for k, v in user_coords.items())
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
        exit(1)

    def __repr__(self, *args, _indent_="\n", **kwargs):
        more = ",".join([str(a)[:5000] for a in args])
        more += ",".join([f"{k}={v}"[:5000] for k, v in kwargs.items()])

        dates = " no-dates"
        if self.group_of_dates is not None:
            dates = f" {len(self.group_of_dates)} dates"
            dates += " ("
            dates += "/".join(d.strftime("%Y-%m-%dT%H:%M") for d in self.group_of_dates)
            if len(dates) > 100:
                dates = dates[:100] + "..."
            dates += ")"

        more = more[:5000]
        txt = f"{self.__class__.__name__}:{dates}{_indent_}{more}"
        if _indent_:
            txt = txt.replace("\n", "\n  ")
        return txt

    def _raise_not_implemented(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.__class__.__name__}({self.group_of_dates})"

    def build_coords(self):
        if self._coords_already_built:
            return

        cube = self.get_cube()

        from_data = cube.user_coords
        from_config = self.context.order_by

        keys_from_config = list(from_config.keys())
        keys_from_data = list(from_data.keys())
        assert keys_from_data == keys_from_config, f"Critical error: {keys_from_data=} != {keys_from_config=}. {self=}"

        variables_key = list(from_config.keys())[1]
        ensembles_key = list(from_config.keys())[2]

        if isinstance(from_config[variables_key], (list, tuple)):
            assert all([v == w for v, w in zip(from_data[variables_key], from_config[variables_key])]), (
                from_data[variables_key],
                from_config[variables_key],
            )

        self._variables = from_data[variables_key]  # "param_level"
        self._ensembles = from_data[ensembles_key]  # "number"

        first_field = self.datasource[0]
        grid_points = first_field.grid_points()

        lats, lons = grid_points

        assert len(lats) == len(lons), (len(lats), len(lons), first_field)
        assert len(lats) == math.prod(first_field.shape), (len(lats), first_field.shape, first_field)

        north = np.amax(lats)
        south = np.amin(lats)
        east = np.amax(lons)
        west = np.amin(lons)

        assert -90 <= south <= north <= 90, (south, north, first_field)
        assert (-180 <= west <= east <= 180) or (0 <= west <= east <= 360), (
            west,
            east,
            first_field,
        )

        grid_values = list(range(len(grid_points[0])))

        self._grid_points = grid_points
        self._resolution = first_field.resolution
        self._grid_values = grid_values
        self._field_shape = first_field.shape
        self._proj_string = first_field.proj_string if hasattr(first_field, "proj_string") else None

        self._cube = cube

        self._coords_already_built = True

    @property
    def variables(self):
        self.build_coords()
        return self._variables

    @property
    def variables_metadata(self):
        return _fields_metatata(self.variables, self._cube)

    @property
    def ensembles(self):
        self.build_coords()
        return self._ensembles

    @property
    def resolution(self):
        self.build_coords()
        return self._resolution

    @property
    def grid_values(self):
        self.build_coords()
        return self._grid_values

    @property
    def grid_points(self):
        self.build_coords()
        return self._grid_points

    @property
    def field_shape(self):
        self.build_coords()
        return self._field_shape

    @property
    def proj_string(self):
        self.build_coords()
        return self._proj_string

    @cached_property
    def shape(self):
        return [
            len(self.group_of_dates),
            len(self.variables),
            len(self.ensembles),
            len(self.grid_values),
        ]

    @cached_property
    def coords(self):
        return {
            "dates": list(self.group_of_dates),
            "variables": self.variables,
            "ensembles": self.ensembles,
            "values": self.grid_values,
        }
