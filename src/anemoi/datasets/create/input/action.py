# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from copy import deepcopy

from anemoi.utils.dates import as_datetime as as_datetime
from anemoi.utils.dates import frequency_to_timedelta as frequency_to_timedelta
from earthkit.data.core.order import build_remapping

from anemoi.datasets.dates import DatesProvider as DatesProvider
from anemoi.datasets.fields import FieldArray as FieldArray
from anemoi.datasets.fields import NewValidDateTimeField as NewValidDateTimeField

from .context import Context
from .misc import is_function

LOG = logging.getLogger(__name__)


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

    def _trace_select(self, group_of_dates):
        return f"{self.__class__.__name__}({group_of_dates})"


class ActionContext(Context):
    def __init__(self, /, order_by, flatten_grid, remapping, use_grib_paramid):
        super().__init__()
        self.order_by = order_by
        self.flatten_grid = flatten_grid
        self.remapping = build_remapping(remapping)
        self.use_grib_paramid = use_grib_paramid


def action_factory(config, context, action_path):

    from .concat import ConcatAction
    from .data_sources import DataSourcesAction
    from .function import FunctionAction
    from .join import JoinAction
    from .pipe import PipeAction
    from .repeated_dates import RepeatedDatesAction

    # from .data_sources import DataSourcesAction

    assert isinstance(context, Context), (type, context)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid input config {config}")
    if len(config) != 1:
        raise ValueError(f"Invalid input config. Expecting dict with only one key, got {list(config.keys())}")

    config = deepcopy(config)
    key = list(config.keys())[0]

    if isinstance(config[key], list):
        args, kwargs = config[key], {}
    elif isinstance(config[key], dict):
        args, kwargs = [], config[key]
    else:
        raise ValueError(f"Invalid input config {config[key]} ({type(config[key])}")

    cls = {
        "data_sources": DataSourcesAction,
        "concat": ConcatAction,
        "join": JoinAction,
        "pipe": PipeAction,
        "function": FunctionAction,
        "repeated_dates": RepeatedDatesAction,
    }.get(key)

    if cls is None:
        if not is_function(key, "sources"):
            raise ValueError(f"Unknown action '{key}' in {config}")
        cls = FunctionAction
        args = [key] + args

    return cls(context, action_path + [key], *args, **kwargs)
