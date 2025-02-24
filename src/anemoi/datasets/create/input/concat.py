# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from copy import deepcopy
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from earthkit.data import FieldList

from anemoi.datasets.dates import DatesProvider

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


class ConcatResult(Result):
    """Represents the result of concatenating multiple results."""

    def __init__(
        self,
        context: object,
        action_path: List[str],
        group_of_dates: GroupOfDates,
        results: List[Result],
        **kwargs: Any,
    ) -> None:
        """Initializes a ConcatResult instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : List[str]
            The action path.
        group_of_dates : GroupOfDates
            The group of dates.
        results : List[Result]
            The list of results.
        kwargs : Any
            Additional keyword arguments.
        """
        super().__init__(context, action_path, group_of_dates)
        self.results = [r for r in results if not r.empty]

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self) -> FieldList:
        """Returns the concatenated datasource from all results."""
        ds = EmptyResult(self.context, self.action_path, self.group_of_dates).datasource
        for i in self.results:
            ds += i.datasource
        return _tidy(ds)

    @property
    def variables(self) -> List[str]:
        """Returns the list of variables, ensuring all results have the same variables."""
        variables = None
        for f in self.results:
            if f.empty:
                continue
            if variables is None:
                variables = f.variables
            assert variables == f.variables, (variables, f.variables)
        assert variables is not None, self.results
        return variables

    def __repr__(self) -> str:
        """Returns a string representation of the ConcatResult instance.

        Returns
        -------
        str
            A string representation of the ConcatResult instance.
        """
        content = "\n".join([str(i) for i in self.results])
        return self._repr(content)


class ConcatAction(Action):
    """Represents an action that concatenates multiple actions based on their dates."""

    def __init__(self, context: object, action_path: List[str], *configs: Dict[str, Any]) -> None:
        """Initializes a ConcatAction instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : List[str]
            The action path.
        configs : Dict[str, Any]
            The configuration dictionaries.
        """
        super().__init__(context, action_path, *configs)
        parts = []
        for i, cfg in enumerate(configs):
            if "dates" not in cfg:
                raise ValueError(f"Missing 'dates' in {cfg}")
            cfg = deepcopy(cfg)
            dates_cfg = cfg.pop("dates")
            assert isinstance(dates_cfg, dict), dates_cfg
            filtering_dates = DatesProvider.from_config(**dates_cfg)
            action = action_factory(cfg, context, action_path + [str(i)])
            parts.append((filtering_dates, action))
        self.parts = parts

    def __repr__(self) -> str:
        """Returns a string representation of the ConcatAction instance.

        Returns
        -------
        str
            A string representation of the ConcatAction instance.
        """
        content = "\n".join([str(i) for i in self.parts])
        return self._repr(content)

    @trace_select
    def select(self, group_of_dates: GroupOfDates) -> Union[ConcatResult, EmptyResult]:
        """Selects the concatenated result for the given group of dates.

        Parameters
        ----------
        group_of_dates : GroupOfDates
            The group of dates.

        Returns
        -------
        Union[ConcatResult, EmptyResult]
            The concatenated result or an empty result.
        """
        from anemoi.datasets.dates.groups import GroupOfDates

        results = []
        for filtering_dates, action in self.parts:
            newdates = GroupOfDates(sorted(set(group_of_dates) & set(filtering_dates)), group_of_dates.provider)
            if newdates:
                results.append(action.select(newdates))
        if not results:
            return EmptyResult(self.context, self.action_path, group_of_dates)

        return ConcatResult(self.context, self.action_path, group_of_dates, results)
