# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from functools import cached_property

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


class JoinResult(Result):
    def __init__(self, context, action_path, group_of_dates, results, **kwargs):
        super().__init__(context, action_path, group_of_dates)
        self.results = [r for r in results if not r.empty]

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self):
        ds = EmptyResult(self.context, self.action_path, self.group_of_dates).datasource
        for i in self.results:
            ds += i.datasource
        return _tidy(ds)

    def __repr__(self):
        content = "\n".join([str(i) for i in self.results])
        return super().__repr__(content)


class JoinAction(Action):
    def __init__(self, context, action_path, *configs):
        super().__init__(context, action_path, *configs)
        self.actions = [action_factory(c, context, action_path + [str(i)]) for i, c in enumerate(configs)]

    def __repr__(self):
        content = "\n".join([str(i) for i in self.actions])
        return super().__repr__(content)

    @trace_select
    def select(self, group_of_dates):
        results = [a.select(group_of_dates) for a in self.actions]
        return JoinResult(self.context, self.action_path, group_of_dates, results)
