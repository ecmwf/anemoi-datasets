# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging

from .action import Action
from .action import action_factory
from .step import step_factory
from .trace import trace_select

LOG = logging.getLogger(__name__)


class PipeAction(Action):
    def __init__(self, context, action_path, *configs):
        super().__init__(context, action_path, *configs)
        assert len(configs) > 1, configs
        current = action_factory(configs[0], context, action_path + ["0"])
        for i, c in enumerate(configs[1:]):
            current = step_factory(c, context, action_path + [str(i + 1)], previous_step=current)
        self.last_step = current

    @trace_select
    def select(self, group_of_dates):
        return self.last_step.select(group_of_dates)

    def __repr__(self):
        return super().__repr__(self.last_step)
