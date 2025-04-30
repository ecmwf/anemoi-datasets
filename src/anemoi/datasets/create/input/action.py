# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List

from earthkit.data.core.order import build_remapping

from ...dates.groups import GroupOfDates
from .context import Context
from .template import substitute

LOG = logging.getLogger(__name__)


class Action:
    """Represents an action to be performed within a given context.

    Attributes
    ----------
    context : ActionContext
        The context in which the action exists.
    kwargs : Dict[str, Any]
        Additional keyword arguments.
    args : Any
        Additional positional arguments.
    action_path : List[str]
        The action path.
    """

    def __init__(
        self, context: "ActionContext", action_path: List[str], /, *args: Any, **kwargs: Dict[str, Any]
    ) -> None:
        """Initialize an Action instance.

        Parameters
        ----------
        context : ActionContext
            The context in which the action exists.
        action_path : List[str]
            The action path.
        args : Any
            Additional positional arguments.
        kwargs : Dict[str, Any]
            Additional keyword arguments.
        """
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
    def _short_str(cls, x: str) -> str:
        """Shorten the string representation if it exceeds 1000 characters.

        Parameters
        ----------
        x : str
            The string to shorten.

        Returns
        -------
        str
            The shortened string.
        """
        x = str(x)
        if len(x) < 1000:
            return x
        return x[:1000] + "..."

    def _repr(self, *args: Any, _indent_: str = "\n", _inline_: str = "", **kwargs: Any) -> str:
        """Generate a string representation of the Action instance.

        Parameters
        ----------
        args : Any
            Additional positional arguments.
        _indent_ : str, optional
            The indentation string, by default "\n".
        _inline_ : str, optional
            The inline string, by default "".
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            The string representation.
        """
        more = ",".join([str(a)[:5000] for a in args])
        more += ",".join([f"{k}={v}"[:5000] for k, v in kwargs.items()])

        more = more[:5000]
        txt = f"{self.__class__.__name__}: {_inline_}{_indent_}{more}"
        if _indent_:
            txt = txt.replace("\n", "\n  ")
        return txt

    def __repr__(self) -> str:
        """Return the string representation of the Action instance.

        Returns
        -------
        str
            The string representation.
        """
        return self._repr()

    def select(self, dates: object, **kwargs: Any) -> None:
        """Select dates for the action.

        Parameters
        ----------
        dates : object
            The dates to select.
        kwargs : Any
            Additional keyword arguments.
        """
        self._raise_not_implemented()

    def _raise_not_implemented(self) -> None:
        """Raise a NotImplementedError indicating the method is not implemented."""
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")

    def _trace_select(self, group_of_dates: GroupOfDates) -> str:
        """Trace the selection of a group of dates.

        Parameters
        ----------
        group_of_dates : GroupOfDates
            The group of dates to trace.

        Returns
        -------
        str
            The trace string.
        """
        return f"{self.__class__.__name__}({group_of_dates})"


class ActionContext(Context):
    """Represents the context in which an action is performed.

    Attributes
    ----------
    order_by : str
        The order by criteria.
    flatten_grid : bool
        Whether to flatten the grid.
    remapping : Dict[str, Any]
        The remapping configuration.
    use_grib_paramid : bool
        Whether to use GRIB parameter ID.
    """

    def __init__(self, /, order_by: str, flatten_grid: bool, remapping: Dict[str, Any], use_grib_paramid: bool) -> None:
        """Initialize an ActionContext instance.

        Parameters
        ----------
        order_by : str
            The order by criteria.
        flatten_grid : bool
            Whether to flatten the grid.
        remapping : Dict[str, Any]
            The remapping configuration.
        use_grib_paramid : bool
            Whether to use GRIB parameter ID.
        """
        super().__init__()
        self.order_by = order_by
        self.flatten_grid = flatten_grid
        self.remapping = build_remapping(remapping)
        self.use_grib_paramid = use_grib_paramid


def action_factory(config: Dict[str, Any], context: ActionContext, action_path: List[str]) -> Action:
    """Factory function to create an Action instance based on the configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        The action configuration.
    context : ActionContext
        The context in which the action exists.
    action_path : List[str]
        The action path.

    Returns
    -------
    Action
        The created Action instance.
    """
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
        print(json.dumps(config, indent=2, default=str))
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
        "data-sources": DataSourcesAction,
        "concat": ConcatAction,
        "join": JoinAction,
        "pipe": PipeAction,
        "function": FunctionAction,
        "repeated_dates": RepeatedDatesAction,
        "repeated-dates": RepeatedDatesAction,
    }.get(key)

    if cls is None:
        from ..sources import create_source

        source = create_source(None, substitute(context, config))
        return FunctionAction(context, action_path + [key], key, source)

    return cls(context, action_path + [key], *args, **kwargs)
