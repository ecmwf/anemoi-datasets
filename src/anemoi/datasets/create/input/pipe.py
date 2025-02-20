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
from typing import Any

from .action import Action
from .action import action_factory
from .step import step_factory
from .trace import trace_select

LOG = logging.getLogger(__name__)


class PipeAction(Action):
    def __init__(self, context: Any, action_path: list, *configs: dict) -> None:
        super().__init__(context, action_path, *configs)
        if len(configs) <= 1:
            raise ValueError(
                f"PipeAction requires at least two actions, got {len(configs)}\n{json.dumps(configs, indent=2)}"
            )

        current: Any = action_factory(configs[0], context, action_path + ["0"])
        for i, c in enumerate(configs[1:]):
            current = step_factory(c, context, action_path + [str(i + 1)], previous_step=current)
        self.last_step: Any = current

    @trace_select
    def select(self, group_of_dates: Any) -> Any:
        return self.last_step.select(group_of_dates)

    def __repr__(self) -> str:
        return f"PipeAction({self.last_step})"
