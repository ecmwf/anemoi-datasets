# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import itertools
import logging
import math
import time
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from functools import wraps

import numpy as np
from anemoi.utils.humanize import seconds_to_human
from anemoi.utils.humanize import shorten_list
from earthkit.data.core.fieldlist import FieldList
from earthkit.data.core.fieldlist import MultiFieldList
from earthkit.data.core.order import build_remapping

from anemoi.datasets.dates import Dates

from .functions import import_function
from .template import Context
from .template import notify_result
from .template import resolve
from .template import substitute
from .trace import trace
from .trace import trace_datasource
from .trace import trace_select

LOG = logging.getLogger(__name__)


def parse_function_name(name):

    if name.endswith("h") and name[:-1].isdigit():

        if "-" in name:
            name, delta = name.split("-")
            sign = -1

        elif "+" in name:
            name, delta = name.split("+")
            sign = 1

        else:
            return name, None

        assert delta[-1] == "h", (name, delta)
        delta = sign * int(delta[:-1])
        return name, delta

    return name, None


def time_delta_to_string(delta):
    assert isinstance(delta, datetime.timedelta), delta
    seconds = delta.total_seconds()
    hours = int(seconds // 3600)
    assert hours * 3600 == seconds, delta
    hours = abs(hours)

    if seconds > 0:
        return f"plus_{hours}h"
    if seconds == 0:
        return ""
    if seconds < 0:
        return f"minus_{hours}h"


def is_function(name, kind):
    name, delta = parse_function_name(name)  # noqa
    try:
        import_function(name, kind)
        return True
    except ImportError as e:
        print(e)
        return False


def assert_fieldlist(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        assert isinstance(result, FieldList), type(result)
        return result

    return wrapper


def assert_is_fieldlist(obj):
    assert isinstance(obj, FieldList), type(obj)


def _data_request(data):
    date = None
    params_levels = defaultdict(set)
    params_steps = defaultdict(set)

    area = grid = None

    for field in data:
        if not hasattr(field, "as_mars"):
            continue

        if date is None:
            date = field.datetime()["valid_time"]

        if field.datetime()["valid_time"] != date:
            continue

        as_mars = field.metadata(namespace="mars")
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

    def sort(old_dic):
        new_dic = {}
        for k, v in old_dic.items():
            new_dic[k] = sorted(list(v))
        return new_dic

    params_steps = sort(params_steps)
    params_levels = sort(params_levels)

    return dict(param_level=params_levels, param_step=params_steps, area=area, grid=grid)


class Action:
    def __init__(self, context, action_path, /, *args, **kwargs):
        if "args" in kwargs and "kwargs" in kwargs:
            """We have:
               args = []
               kwargs = {args: [...], kwargs: {...}}
            move the content of kwargs to args and kwargs.
            """
            assert len(kwargs) == 2, (args, kwargs)
            assert not args, (args, kwargs)
            args = kwargs.pop("args")
            kwargs = kwargs.pop("kwargs")

        assert isinstance(context, ActionContext), type(context)
        self.context = context
        self.kwargs = kwargs
        self.args = args
        self.action_path = action_path

    @classmethod
    def _short_str(cls, x):
        x = str(x)
        if len(x) < 1000:
            return x
        return x[:1000] + "..."

    def __repr__(self, *args, _indent_="\n", _inline_="", **kwargs):
        more = ",".join([str(a)[:5000] for a in args])
        more += ",".join([f"{k}={v}"[:5000] for k, v in kwargs.items()])

        more = more[:5000]
        txt = f"{self.__class__.__name__}: {_inline_}{_indent_}{more}"
        if _indent_:
            txt = txt.replace("\n", "\n  ")
        return txt

    def select(self, dates, **kwargs):
        self._raise_not_implemented()

    def _raise_not_implemented(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")

    def _trace_select(self, dates):
        return f"{self.__class__.__name__}({shorten(dates)})"


def shorten(dates):
    if isinstance(dates, (list, tuple)):
        dates = [d.isoformat() for d in dates]
        if len(dates) > 5:
            return f"{dates[0]}...{dates[-1]}"
    return dates


class Result:
    empty = False
    _coords_already_built = False

    def __init__(self, context, action_path, dates):
        assert isinstance(context, ActionContext), type(context)
        assert isinstance(action_path, list), action_path

        self.context = context
        self.dates = dates
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

        names = []
        for a in args:
            if isinstance(a, str):
                names.append(a)
            elif isinstance(a, dict):
                names += list(a.keys())

        print(f"Building a {len(names)}D hypercube using", names)

        ds = ds.order_by(*args, remapping=remapping, patches=patches)
        user_coords = ds.unique_values(*names, remapping=remapping, patches=patches)

        print()
        print("Number of unique values found for each coordinate:")
        for k, v in user_coords.items():
            print(f"  {k:20}:", len(v))
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

        if math.prod(user_shape) > len(ds):
            print(f"This means that all the fields in the datasets do not exists for all combinations of {names}.")

            for f in ds:
                metadata = remapping(f.metadata)
                expected.remove(tuple(metadata(n) for n in names))

            print("Missing fields:")
            print()
            for i, f in enumerate(sorted(expected)):
                print(" ", f)
                if i >= 9 and len(expected) > 10:
                    print("...", len(expected) - i - 1, "more")
                    break

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
        if self.dates is not None:
            dates = f" {len(self.dates)} dates"
            dates += " ("
            dates += "/".join(d.strftime("%Y-%m-%d:%H") for d in self.dates)
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
        return f"{self.__class__.__name__}({shorten(self.dates)})"

    def build_coords(self):
        if self._coords_already_built:
            return
        from_data = self.get_cube().user_coords
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

    @property
    def variables(self):
        self.build_coords()
        return self._variables

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
            len(self.dates),
            len(self.variables),
            len(self.ensembles),
            len(self.grid_values),
        ]

    @cached_property
    def coords(self):
        return {
            "dates": self.dates,
            "variables": self.variables,
            "ensembles": self.ensembles,
            "values": self.grid_values,
        }


class EmptyResult(Result):
    empty = True

    def __init__(self, context, action_path, dates):
        super().__init__(context, action_path + ["empty"], dates)

    @cached_property
    @assert_fieldlist
    @trace_datasource
    def datasource(self):
        from earthkit.data import from_source

        return from_source("empty")

    @property
    def variables(self):
        return []


def _flatten(ds):
    if isinstance(ds, MultiFieldList):
        return [_tidy(f) for s in ds._indexes for f in _flatten(s)]
    return [ds]


def _tidy(ds, indent=0):
    if isinstance(ds, MultiFieldList):

        sources = [s for s in _flatten(ds) if len(s) > 0]
        if len(sources) == 1:
            return sources[0]
        return MultiFieldList(sources)
    return ds


class FunctionResult(Result):
    def __init__(self, context, action_path, dates, action):
        super().__init__(context, action_path, dates)
        assert isinstance(action, Action), type(action)
        self.action = action

        self.args, self.kwargs = substitute(context, (self.action.args, self.action.kwargs))

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.action.name}({shorten(self.dates)})"

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self):
        args, kwargs = resolve(self.context, (self.args, self.kwargs))

        try:
            return _tidy(self.action.function(FunctionContext(self), self.dates, *args, **kwargs))
        except Exception:
            LOG.error(f"Error in {self.action.function.__name__}", exc_info=True)
            raise

    def __repr__(self):
        try:
            return f"{self.action.name}({shorten(self.dates)})"
        except Exception:
            return f"{self.__class__.__name__}(unitialised)"

    @property
    def function(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")


class JoinResult(Result):
    def __init__(self, context, action_path, dates, results, **kwargs):
        super().__init__(context, action_path, dates)
        self.results = [r for r in results if not r.empty]

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self):
        ds = EmptyResult(self.context, self.action_path, self.dates).datasource
        for i in self.results:
            ds += i.datasource
        return _tidy(ds)

    def __repr__(self):
        content = "\n".join([str(i) for i in self.results])
        return super().__repr__(content)


class DateShiftAction(Action):
    def __init__(self, context, action_path, delta, **kwargs):
        super().__init__(context, action_path, **kwargs)

        if isinstance(delta, str):
            if delta[0] == "-":
                delta, sign = int(delta[1:]), -1
            else:
                delta, sign = int(delta), 1
            delta = datetime.timedelta(hours=sign * delta)
        assert isinstance(delta, int), delta
        delta = datetime.timedelta(hours=delta)
        self.delta = delta

        self.content = action_factory(kwargs, context, self.action_path + ["shift"])

    @trace_select
    def select(self, dates):
        shifted_dates = [d + self.delta for d in dates]
        result = self.content.select(shifted_dates)
        return UnShiftResult(self.context, self.action_path, dates, result, action=self)

    def __repr__(self):
        return super().__repr__(f"{self.delta}\n{self.content}")


class UnShiftResult(Result):
    def __init__(self, context, action_path, dates, result, action):
        super().__init__(context, action_path, dates)
        # dates are the actual requested dates
        # result does not have the same dates
        self.action = action
        self.result = result

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.action.delta}({shorten(self.dates)})"

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self):
        from earthkit.data.indexing.fieldlist import FieldArray

        class DateShiftedField:
            def __init__(self, field, delta):
                self.field = field
                self.delta = delta

            def metadata(self, key):
                value = self.field.metadata(key)
                if key == "param":
                    return value + "_" + time_delta_to_string(self.delta)
                if key == "valid_datetime":
                    dt = datetime.datetime.fromisoformat(value)
                    new_dt = dt - self.delta
                    new_value = new_dt.isoformat()
                    return new_value
                if key in ["date", "time", "step", "hdate"]:
                    raise NotImplementedError(f"metadata {key} not implemented when shifting dates")
                return value

            def __getattr__(self, name):
                return getattr(self.field, name)

        ds = self.result.datasource
        ds = FieldArray([DateShiftedField(fs, self.action.delta) for fs in ds])
        return _tidy(ds)


class FunctionAction(Action):
    def __init__(self, context, action_path, _name, **kwargs):
        super().__init__(context, action_path, **kwargs)
        self.name = _name

    @trace_select
    def select(self, dates):
        return FunctionResult(self.context, self.action_path, dates, action=self)

    @property
    def function(self):
        # name, delta = parse_function_name(self.name)
        return import_function(self.name, "sources")

    def __repr__(self):
        content = ""
        content += ",".join([self._short_str(a) for a in self.args])
        content += " ".join([self._short_str(f"{k}={v}") for k, v in self.kwargs.items()])
        content = self._short_str(content)
        return super().__repr__(_inline_=content, _indent_=" ")

    def _trace_select(self, dates):
        return f"{self.name}({shorten(dates)})"


class PipeAction(Action):
    def __init__(self, context, action_path, *configs):
        super().__init__(context, action_path, *configs)
        assert len(configs) > 1, configs
        current = action_factory(configs[0], context, action_path + ["0"])
        for i, c in enumerate(configs[1:]):
            current = step_factory(c, context, action_path + [str(i + 1)], previous_step=current)
        self.last_step = current

    @trace_select
    def select(self, dates):
        return self.last_step.select(dates)

    def __repr__(self):
        return super().__repr__(self.last_step)


class StepResult(Result):
    def __init__(self, context, action_path, dates, action, upstream_result):
        super().__init__(context, action_path, dates)
        assert isinstance(upstream_result, Result), type(upstream_result)
        self.upstream_result = upstream_result
        self.action = action

    @property
    @notify_result
    @trace_datasource
    def datasource(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")


class StepAction(Action):
    result_class = None

    def __init__(self, context, action_path, previous_step, *args, **kwargs):
        super().__init__(context, action_path, *args, **kwargs)
        self.previous_step = previous_step

    @trace_select
    def select(self, dates):
        return self.result_class(
            self.context,
            self.action_path,
            dates,
            self,
            self.previous_step.select(dates),
        )

    def __repr__(self):
        return super().__repr__(self.previous_step, _inline_=str(self.kwargs))


class StepFunctionResult(StepResult):
    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self):
        try:
            return _tidy(
                self.action.function(
                    FunctionContext(self),
                    self.upstream_result.datasource,
                    *self.action.args[1:],
                    **self.action.kwargs,
                )
            )

        except Exception:
            LOG.error(f"Error in {self.action.name}", exc_info=True)
            raise

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.action.name}({shorten(self.dates)})"


class FilterStepResult(StepResult):
    @property
    @notify_result
    @assert_fieldlist
    @trace_datasource
    def datasource(self):
        ds = self.upstream_result.datasource
        ds = ds.sel(**self.action.kwargs)
        return _tidy(ds)


class FilterStepAction(StepAction):
    result_class = FilterStepResult


class FunctionStepAction(StepAction):
    result_class = StepFunctionResult

    def __init__(self, context, action_path, previous_step, *args, **kwargs):
        super().__init__(context, action_path, previous_step, *args, **kwargs)
        self.name = args[0]
        self.function = import_function(self.name, "filters")


class ConcatResult(Result):
    def __init__(self, context, action_path, dates, results, **kwargs):
        super().__init__(context, action_path, dates)
        self.results = [r for r in results if not r.empty]

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self):
        ds = EmptyResult(self.context, self.action_path, self.dates).datasource
        for i in self.results:
            ds += i.datasource
        return _tidy(ds)

    @property
    def variables(self):
        """Check that all the results objects have the same variables."""
        variables = None
        for f in self.results:
            if f.empty:
                continue
            if variables is None:
                variables = f.variables
            assert variables == f.variables, (variables, f.variables)
        assert variables is not None, self.results
        return variables

    def __repr__(self):
        content = "\n".join([str(i) for i in self.results])
        return super().__repr__(content)


class DataSourcesResult(Result):
    def __init__(self, context, action_path, dates, input_result, sources_results):
        super().__init__(context, action_path, dates)
        # result is the main input result
        self.input_result = input_result
        # sources_results is the list of the sources_results
        self.sources_results = sources_results

    @cached_property
    def datasource(self):
        for i in self.sources_results:
            # for each result trigger the datasource to be computed
            # and saved in context
            self.context.notify_result(i.action_path[:-1], i.datasource)
        # then return the input result
        # which can use the datasources of the included results
        return _tidy(self.input_result.datasource)


class DataSourcesAction(Action):
    def __init__(self, context, action_path, sources, input):
        super().__init__(context, ["data_sources"], *sources)
        if isinstance(sources, dict):
            configs = [(str(k), c) for k, c in sources.items()]
        elif isinstance(sources, list):
            configs = [(str(i), c) for i, c in enumerate(sources)]
        else:
            raise ValueError(f"Invalid data_sources, expecting list or dict, got {type(sources)}: {sources}")

        self.sources = [action_factory(config, context, ["data_sources"] + [a_path]) for a_path, config in configs]
        self.input = action_factory(input, context, ["input"])

    def select(self, dates):
        sources_results = [a.select(dates) for a in self.sources]
        return DataSourcesResult(
            self.context,
            self.action_path,
            dates,
            self.input.select(dates),
            sources_results,
        )

    def __repr__(self):
        content = "\n".join([str(i) for i in self.sources])
        return super().__repr__(content)


class ConcatAction(Action):
    def __init__(self, context, action_path, *configs):
        super().__init__(context, action_path, *configs)
        parts = []
        for i, cfg in enumerate(configs):
            if "dates" not in cfg:
                raise ValueError(f"Missing 'dates' in {cfg}")
            cfg = deepcopy(cfg)
            dates_cfg = cfg.pop("dates")
            assert isinstance(dates_cfg, dict), dates_cfg
            filtering_dates = Dates.from_config(**dates_cfg)
            action = action_factory(cfg, context, action_path + [str(i)])
            parts.append((filtering_dates, action))
        self.parts = parts

    def __repr__(self):
        content = "\n".join([str(i) for i in self.parts])
        return super().__repr__(content)

    @trace_select
    def select(self, dates):
        results = []
        for filtering_dates, action in self.parts:
            newdates = sorted(set(dates) & set(filtering_dates))
            if newdates:
                results.append(action.select(newdates))
        if not results:
            return EmptyResult(self.context, self.action_path, dates)

        return ConcatResult(self.context, self.action_path, dates, results)


class JoinAction(Action):
    def __init__(self, context, action_path, *configs):
        super().__init__(context, action_path, *configs)
        self.actions = [action_factory(c, context, action_path + [str(i)]) for i, c in enumerate(configs)]

    def __repr__(self):
        content = "\n".join([str(i) for i in self.actions])
        return super().__repr__(content)

    @trace_select
    def select(self, dates):
        results = [a.select(dates) for a in self.actions]
        return JoinResult(self.context, self.action_path, dates, results)


def action_factory(config, context, action_path):
    assert isinstance(context, Context), (type, context)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid input config {config}")
    if len(config) != 1:
        raise ValueError(f"Invalid input config. Expecting dict with only one key, got {list(config.keys())}")

    config = deepcopy(config)
    key = list(config.keys())[0]

    if isinstance(config[key], list):
        args, kwargs = config[key], {}
    if isinstance(config[key], dict):
        args, kwargs = [], config[key]

    cls = {
        # "date_shift": DateShiftAction,
        # "date_filter": DateFilterAction,
        "data_sources": DataSourcesAction,
        "concat": ConcatAction,
        "join": JoinAction,
        "pipe": PipeAction,
        "function": FunctionAction,
    }.get(key)

    if cls is None:
        if not is_function(key, "sources"):
            raise ValueError(f"Unknown action '{key}' in {config}")
        cls = FunctionAction
        args = [key] + args

    return cls(context, action_path + [key], *args, **kwargs)


def step_factory(config, context, action_path, previous_step):
    assert isinstance(context, Context), (type, context)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid input config {config}")

    config = deepcopy(config)
    assert len(config) == 1, config

    key = list(config.keys())[0]
    cls = dict(
        filter=FilterStepAction,
        # rename=RenameAction,
        # remapping=RemappingAction,
    ).get(key)

    if isinstance(config[key], list):
        args, kwargs = config[key], {}

    if isinstance(config[key], dict):
        args, kwargs = [], config[key]

    if isinstance(config[key], str):
        args, kwargs = [config[key]], {}

    if cls is None:
        if not is_function(key, "filters"):
            raise ValueError(f"Unknown step {key}")
        cls = FunctionStepAction
        args = [key] + args
        # print("========", args)

    return cls(context, action_path, previous_step, *args, **kwargs)


class FunctionContext:
    """A FunctionContext is passed to all functions, it will be used to pass information
    to the functions from the other actions and filters and results.
    """

    def __init__(self, owner):
        self.owner = owner
        self.use_grib_paramid = owner.context.use_grib_paramid

    def trace(self, emoji, *args):
        trace(emoji, *args)


class ActionContext(Context):
    def __init__(self, /, order_by, flatten_grid, remapping, use_grib_paramid):
        super().__init__()
        self.order_by = order_by
        self.flatten_grid = flatten_grid
        self.remapping = build_remapping(remapping)
        self.use_grib_paramid = use_grib_paramid


class InputBuilder:
    def __init__(self, config, data_sources, **kwargs):
        self.kwargs = kwargs

        config = deepcopy(config)
        if data_sources:
            config = dict(
                data_sources=dict(
                    sources=data_sources,
                    input=config,
                )
            )
        self.config = config
        self.action_path = ["input"]

    @trace_select
    def select(self, dates):
        """This changes the context."""
        context = ActionContext(**self.kwargs)
        action = action_factory(self.config, context, self.action_path)
        return action.select(dates)

    def __repr__(self):
        context = ActionContext(**self.kwargs)
        a = action_factory(self.config, context, self.action_path)
        return repr(a)

    def _trace_select(self, dates):
        return f"InputBuilder({shorten(dates)})"


build_input = InputBuilder
