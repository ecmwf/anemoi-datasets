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
from typing import List

from earthkit.data import FieldList

from ..sources.accumulations2 import accumulations
from ...dates.groups import GroupOfDates
from .action import Action
from .action import action_factory
from .empty import EmptyResult
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
        self, context: object, action_path: list, group_of_dates: GroupOfDates, source: Any, accumulate: Any, **kwargs: Any
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
        accumulate: Any
            The accumulate operation to be used (as action_factory)
        """
        super().__init__(context, action_path, group_of_dates)
        self.source: Any = source
        self.accumulate: Any = accumulate
        
    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self) -> FieldList:
        """Returns the combined datasource from all results."""
        
        ds = accumulations(self.context, self.group_of_dates, self.source, self.accumulate['accumulations'])
        
        return _tidy(ds)

    def __repr__(self) -> str:
        """Returns a string representation of the JoinResult instance."""
        content: str = "\n".join([str(i) for i in self.results])
        return self._repr(content)

class AccumulationAction(Action):
    """An Action implementation that selects and transforms a group of dates."""

    def __init__(self, context: Any, action_path: List[str], source: Any, **kwargs: Any) -> None:
        """Initialize RepeatedDatesAction.

        Args:
            context (Any): The context.
            action_path (List[str]): The action path.
            source (Any): The data source.
            mode (str): The mode for date mapping.
            **kwargs (Any): Additional arguments.
        """
        super().__init__(context, action_path, source)

        self.source: Any = action_factory(source, context, action_path + ["source"])
        self.accumulate: Any = action_factory({'accumulations' : kwargs}, context, action_path)

    @trace_select
    def select(self, group_of_dates: Any) -> AccumulateResult:
        """Select and transform the group of dates.

        Args:
            group_of_dates (Any): The group of dates to select.

        Returns
        -------
        AccumulationResult
            The result of the accumulate operation.
        """
        
        return AccumulationResult(self.context, self.action_path, group_of_dates, self.source, self.accumulate)