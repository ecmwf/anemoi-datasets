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
from typing import List
from typing import Union

from earthkit.data import FieldList

from anemoi.datasets.dates import DatesProvider

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
    def __init__(
        self,
        context: object,
        action_path: list,
        group_of_dates: object,
        results: List[Result],
        **kwargs,
    ) -> None:
        super().__init__(context, action_path, group_of_dates)
        self.results = [r for r in results if not r.empty]

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self) -> FieldList:
        ds = EmptyResult(self.context, self.action_path, self.group_of_dates).datasource
        for i in self.results:
            ds += i.datasource
        return _tidy(ds)

    @property
    def variables(self) -> List[str]:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.results})"


class ConcatAction(Action):
    def __init__(self, context: object, action_path: list, *configs: dict) -> None:
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
        content = "\n".join([str(i) for i in self.parts])
        return super().__repr__(content)

    @trace_select
    def select(self, group_of_dates: object) -> Union[ConcatResult, EmptyResult]:
        from anemoi.datasets.dates.groups import GroupOfDates

        results = []
        for filtering_dates, action in self.parts:
            newdates = GroupOfDates(sorted(set(group_of_dates) & set(filtering_dates)), group_of_dates.provider)
            if newdates:
                results.append(action.select(newdates))
        if not results:
            return EmptyResult(self.context, self.action_path, group_of_dates)

        return ConcatResult(self.context, self.action_path, group_of_dates, results)
