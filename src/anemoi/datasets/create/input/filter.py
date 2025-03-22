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
from typing import Type

from earthkit.data import FieldList

from .function import FunctionContext
from .misc import _tidy
from .misc import assert_fieldlist
from .step import StepAction
from .step import StepResult
from .template import notify_result
from .trace import trace_datasource

LOG = logging.getLogger(__name__)


class FilterStepResult(StepResult):
    @property
    @notify_result
    @assert_fieldlist
    @trace_datasource
    def datasource(self) -> FieldList:
        """Returns the filtered datasource."""
        ds: FieldList = self.upstream_result.datasource
        ds = ds.sel(**self.action.kwargs)
        return _tidy(ds)


class FilterStepAction(StepAction):
    """Represents an action to filter a step result."""

    result_class: Type[FilterStepResult] = FilterStepResult


class StepFunctionResult(StepResult):
    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self) -> FieldList:
        """Returns the datasource after applying the function."""

        self.action.filter.context = FunctionContext(self)
        try:
            return _tidy(
                self.action.filter.execute(
                    self.upstream_result.datasource,
                    *self.action.args[1:],
                    **self.action.kwargs,
                )
            )

        except Exception:
            LOG.error(f"Error in {self.action.name}", exc_info=True)
            raise

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
            A string representation of the traced datasource.
        """
        return f"{self.action.name}({self.group_of_dates})"


class FunctionStepAction(StepAction):
    """Represents an action to apply a function to a step result."""

    result_class: Type[StepFunctionResult] = StepFunctionResult

    def __init__(
        self,
        context: object,
        action_path: list,
        previous_step: StepAction,
        name: str,
        filter: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes a FunctionStepAction instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : list
            The action path.
        previous_step : StepAction
            The previous step action.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, action_path, previous_step, *args, **kwargs)
        self.name = name
        self.filter = filter
