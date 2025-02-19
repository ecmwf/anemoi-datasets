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
from typing import Any
from typing import Union

from ..dates import GroupOfDates
from .trace import trace_select

LOG = logging.getLogger(__name__)


class InputBuilder:
    def __init__(self, config: dict, data_sources: Union[dict, list], **kwargs: Any) -> None:
        self.kwargs = kwargs

        config = deepcopy(config)
        if data_sources:
            config = dict(
                data_sources=dict(
                    sources=data_sources,
                    input=config,
                )
            )
        self.config = config
        self.action_path = ["input"]

    @trace_select
    def select(self, group_of_dates: GroupOfDates) -> Any:
        from .action import ActionContext
        from .action import action_factory

        """This changes the context."""
        context = ActionContext(**self.kwargs)
        action = action_factory(self.config, context, self.action_path)
        return action.select(group_of_dates)

    def __repr__(self) -> str:
        from .action import ActionContext
        from .action import action_factory

        context = ActionContext(**self.kwargs)
        a = action_factory(self.config, context, self.action_path)
        return repr(a)

    def _trace_select(self, group_of_dates: GroupOfDates) -> str:
        return f"InputBuilder({group_of_dates})"


def build_input(config: dict, data_sources: Union[dict, list], **kwargs: Any) -> InputBuilder:
    return InputBuilder(config, data_sources, **kwargs)
