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
from typing import List

from earthkit.data import FieldList

from ...dates.groups import GroupOfDates
from ..sources.accumulations2 import accumulations
from .action import Action
from .action import action_factory
from .misc import _tidy
from .misc import assert_fieldlist
from .result import Result
from .template import notify_result
from .trace import trace_datasource
from .trace import trace_select

LOG = logging.getLogger(__name__)


class AccumulationResult(Result):
    """Represents a result that accumulates multiple fields.

    Attributes
    ----------
    context : object
        The context object.
    action_path : list
        The action path.
    group_of_dates : GroupOfDates
        The group of dates.
    results : List[Result]
        The list of results.
    """

    def __init__(
        self,
        context: object,
        action_path: list,
        group_of_dates: GroupOfDates,
        source: Any,
        request: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Initializes a AccumulationResult instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : list
            The action path.
        group_of_dates : GroupOfDates
            The group of dates.
        source: Any
            The original source used to perform accumulation (as action_factory)
        request: Dict[str,Any]
            The description of the accumulate request
        """
        super().__init__(context, action_path, group_of_dates)
        self.source: Any = source
        self.request: Dict[str, Any] = request

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self) -> FieldList:
        """Returns the combined datasource from all results."""
        ds = accumulations(self.context, self.group_of_dates, self.source, **self.request)

        return _tidy(ds)

    def __repr__(self) -> str:
        """Returns a string representation of the AccumulationResult instance."""
        content: str = f"AccumulationRsult({self.source})"
        return self._repr(content)


class AccumulationAction(Action):
    """An Action implementation that selects and transforms a group of dates."""

    def __init__(self, context: Any, action_path: List[str], source: Dict[str, Any]) -> None:
        """Initialize AccumulationAction.

        Parameters
        ----------
            context: Any
                The context needed to initialize the action
            action_path: List[str]
                The action path to initialize the action.
            source: Dict[str, Any]
                The configuration describing the data source.
        """
        super().__init__(context, action_path, source)

        self.source: Any = action_factory(source, context, action_path + ["source"])
        self.request = source[list(source.keys())[0]]

    @trace_select
    def select(self, group_of_dates: GroupOfDates) -> AccumulationResult:
        """Select and transform the group of dates.

        Parameters
        ----------
            group_of_dates: GroupOfDates
                The group of dates to select.

        Returns
        -------
        AccumulationResult
            The result of the accumulate operation.
        """

        return AccumulationResult(self.context, self.action_path, group_of_dates, self.source, self.request)
