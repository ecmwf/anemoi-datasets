# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from typing import Any
from typing import Dict

from earthkit.data import FieldList

from ...dates.groups import GroupOfDates
from .action import Action
from .misc import _tidy
from .misc import assert_fieldlist
from .result import Result
from .template import notify_result
from .template import substitute
from .trace import trace
from .trace import trace_datasource
from .trace import trace_select

LOG = logging.getLogger(__name__)


class FunctionContext:
    """A FunctionContext is passed to all functions, it will be used to pass information
    to the functions from the other actions and filters and results.
    """

    def __init__(self, owner: Result) -> None:
        """Initializes a FunctionContext instance.

        Parameters
        ----------
        owner : object
            The owner object.
        """
        self.owner = owner
        self.use_grib_paramid: bool = owner.context.use_grib_paramid

    def trace(self, emoji: str, *args: Any) -> None:
        """Traces the given arguments with an emoji.

        Parameters
        ----------
        emoji : str
            The emoji to use.
        *args : Any
            The arguments to trace.
        """
        trace(emoji, *args)

    def info(self, *args: Any, **kwargs: Any) -> None:
        """Logs an info message.

        Parameters
        ----------
        *args : Any
            The arguments for the log message.
        **kwargs : Any
            The keyword arguments for the log message.
        """
        LOG.info(*args, **kwargs)

    @property
    def dates_provider(self) -> object:
        """Returns the dates provider."""
        return self.owner.group_of_dates.provider

    @property
    def partial_ok(self) -> bool:
        """Returns whether partial results are acceptable."""
        return self.owner.group_of_dates.partial_ok

    def get_result(self, *args, **kwargs) -> Any:
        return self.owner.context.get_result(*args, **kwargs)


class FunctionAction(Action):
    """Represents an action that executes a function.

    Attributes
    ----------
    name : str
        The name of the function.
    """

    def __init__(self, context: object, action_path: list, _name: str, source, **kwargs: Dict[str, Any]) -> None:
        """Initializes a FunctionAction instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : list
            The action path.
        _name : str
            The name of the function.
        **kwargs : Dict[str, Any]
            Additional keyword arguments.
        """
        super().__init__(context, action_path, **kwargs)
        self.name: str = _name
        self.source = source

    @trace_select
    def select(self, group_of_dates: GroupOfDates) -> "FunctionResult":
        """Selects the function result for the given group of dates.

        Parameters
        ----------
        group_of_dates : GroupOfDates
            The group of dates.

        Returns
        -------
        FunctionResult
            The function result instance.
        """
        return FunctionResult(self.context, self.action_path, group_of_dates, action=self)

    def __repr__(self) -> str:
        """Returns a string representation of the FunctionAction instance."""
        content: str = ""
        content += ",".join([self._short_str(a) for a in self.args])
        content += " ".join([self._short_str(f"{k}={v}") for k, v in self.kwargs.items()])
        content = self._short_str(content)
        return self._repr(_inline_=content, _indent_=" ")

    def _trace_select(self, group_of_dates: GroupOfDates) -> str:
        """Traces the selection of the function for the given group of dates.

        Parameters
        ----------
        group_of_dates : GroupOfDates
            The group of dates.

        Returns
        -------
        str
            The trace string.
        """
        return f"{self.name}({group_of_dates})"


class FunctionResult(Result):
    """Represents the result of executing a function.

    Attributes
    ----------
    action : Action
        The action instance.
    args : tuple
        The positional arguments for the function.
    kwargs : dict
        The keyword arguments for the function.
    """

    def __init__(self, context: object, action_path: list, group_of_dates: GroupOfDates, action: Action) -> None:
        """Initializes a FunctionResult instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : list
            The action path.
        group_of_dates : GroupOfDates
            The group of dates.
        action : Action
            The action instance.
        """
        super().__init__(context, action_path, group_of_dates)
        assert isinstance(action, Action), type(action)
        self.action: Action = action

        self.args, self.kwargs = substitute(context, (self.action.args, self.action.kwargs))

    def _trace_datasource(self, *args: Any, **kwargs: Any) -> str:
        """Traces the datasource for the given arguments.

        Parameters
        ----------
        *args : Any
            The arguments.
        **kwargs : Any
            The keyword arguments.

        Returns
        -------
        str
            The trace string.
        """
        return f"{self.action.name}({self.group_of_dates})"

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self) -> FieldList:
        """Returns the datasource for the function result."""
        # args, kwargs = resolve(self.context, (self.args, self.kwargs))
        self.action.source.context = FunctionContext(self)

        return _tidy(
            self.action.source.execute(
                list(self.group_of_dates),  # Will provide a list of datetime objects
            )
        )

    def __repr__(self) -> str:
        """Returns a string representation of the FunctionResult instance."""
        try:
            return f"{self.action.name}({self.group_of_dates})"
        except Exception:
            return f"{self.__class__.__name__}(unitialised)"

    @property
    def function(self) -> None:
        """Raises NotImplementedError as this property is not implemented.

        Raises
        ------
        NotImplementedError
            Always raised.
        """
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")
