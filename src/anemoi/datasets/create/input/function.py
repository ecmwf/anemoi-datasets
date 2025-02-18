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
from typing import Callable
from typing import Dict

from earthkit.data import FieldList

from ..functions import import_function
from .action import Action
from .misc import _tidy
from .misc import assert_fieldlist
from .result import Result
from .template import notify_result
from .template import resolve
from .template import substitute
from .trace import trace
from .trace import trace_datasource
from .trace import trace_select

LOG = logging.getLogger(__name__)


class FunctionContext:
    """A FunctionContext is passed to all functions, it will be used to pass information
    to the functions from the other actions and filters and results.
    """

    def __init__(self, owner: object) -> None:
        self.owner = owner
        self.use_grib_paramid = owner.context.use_grib_paramid

    def trace(self, emoji: str, *args: Any) -> None:
        trace(emoji, *args)

    def info(self, *args: Any, **kwargs: Any) -> None:
        LOG.info(*args, **kwargs)

    @property
    def dates_provider(self) -> object:
        return self.owner.group_of_dates.provider

    @property
    def partial_ok(self) -> bool:
        return self.owner.group_of_dates.partial_ok


class FunctionAction(Action):
    def __init__(self, context: object, action_path: list, _name: str, **kwargs: Dict[str, Any]) -> None:
        super().__init__(context, action_path, **kwargs)
        self.name = _name

    @trace_select
    def select(self, group_of_dates: object) -> "FunctionResult":
        return FunctionResult(self.context, self.action_path, group_of_dates, action=self)

    @property
    def function(self) -> Callable:
        return import_function(self.name, "sources")

    def __repr__(self) -> str:
        content = ""
        content += ",".join([self._short_str(a) for a in self.args])
        content += " ".join([self._short_str(f"{k}={v}") for k, v in self.kwargs.items()])
        content = self._short_str(content)
        return super().__repr__(_inline_=content, _indent_=" ")

    def _trace_select(self, group_of_dates: object) -> str:
        return f"{self.name}({group_of_dates})"


class FunctionResult(Result):
    def __init__(self, context: object, action_path: list, group_of_dates: object, action: Action) -> None:
        super().__init__(context, action_path, group_of_dates)
        assert isinstance(action, Action), type(action)
        self.action = action

        self.args, self.kwargs = substitute(context, (self.action.args, self.action.kwargs))

    def _trace_datasource(self, *args: Any, **kwargs: Any) -> str:
        return f"{self.action.name}({self.group_of_dates})"

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self) -> FieldList:
        args, kwargs = resolve(self.context, (self.args, self.kwargs))

        try:
            return _tidy(
                self.action.function(
                    FunctionContext(self),
                    list(self.group_of_dates),  # Will provide a list of datetime objects
                    *args,
                    **kwargs,
                )
            )
        except Exception:
            LOG.error(f"Error in {self.action.function.__name__}", exc_info=True)
            raise

    def __repr__(self) -> str:
        try:
            return f"{self.action.name}({self.group_of_dates})"
        except Exception:
            return f"{self.__class__.__name__}(unitialised)"

    @property
    def function(self) -> None:
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")
