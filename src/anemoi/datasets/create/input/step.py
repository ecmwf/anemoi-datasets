# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from .action import Action
from .action import ActionContext
from .context import Context
from .result import Result
from .template import notify_result
from .trace import trace_datasource
from .trace import trace_select

LOG = logging.getLogger(__name__)


class StepResult(Result):
    """Represents the result of a step in the data processing pipeline."""

    def __init__(
        self, context: Context, action_path: List[str], group_of_dates: Any, action: Action, upstream_result: Result
    ) -> None:
        """Initialize a StepResult instance.

        Parameters
        ----------
        context
            The context in which the step is executed.
        action_path
            The path of actions leading to this step.
        group_of_dates
            The group of dates associated with this step.
        action
            The action associated with this step.
        upstream_result
            The result of the upstream step.
        """
        super().__init__(context, action_path, group_of_dates)
        assert isinstance(upstream_result, Result), type(upstream_result)
        self.upstream_result: Result = upstream_result
        self.action: Action = action

    @property
    @notify_result
    @trace_datasource
    def datasource(self) -> Any:
        """Retrieve the datasource associated with this step result."""
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")


class StepAction(Action):
    """Represents an action that is part of a step in the data processing pipeline."""

    result_class: Optional[Type[StepResult]] = None

    def __init__(
        self, context: ActionContext, action_path: List[str], previous_step: Any, *args: Any, **kwargs: Any
    ) -> None:
        """Initialize a StepAction instance.

        Parameters
        ----------
        context
            The context in which the action is executed.
        action_path
            The path of actions leading to this step.
        previous_step
            The previous step in the pipeline.
        """
        super().__init__(context, action_path, *args, **kwargs)
        self.previous_step: Any = previous_step

    @trace_select
    def select(self, group_of_dates: Any) -> StepResult:
        """Select the result for a given group of dates.

        Parameters
        ----------
        group_of_dates
            The group of dates to select the result for.

        Returns
        -------
        unknown
            The result of the step.
        """
        return self.result_class(
            self.context,
            self.action_path,
            group_of_dates,
            self,
            self.previous_step.select(group_of_dates),
        )

    def __repr__(self) -> str:
        """Return a string representation of the StepAction instance.

        Returns
        -------
        unknown
            String representation of the instance.
        """
        return self._repr(self.previous_step, _inline_=str(self.kwargs))


def step_factory(config: Dict[str, Any], context: ActionContext, action_path: List[str], previous_step: Any) -> Any:
    """Factory function to create a step action based on the given configuration.

    Parameters
    ----------
    config
        The configuration dictionary for the step.
    context
        The context in which the step is executed.
    action_path
        The path of actions leading to this step.
    previous_step
        The previous step in the pipeline.

    Returns
    -------
    unknown
        An instance of a step action.
    """

    from .filter import FilterStepAction
    from .filter import FunctionStepAction

    assert isinstance(context, Context), (type, context)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid input config {config}")

    config = deepcopy(config)
    assert len(config) == 1, config

    key = list(config.keys())[0]
    cls = dict(
        filter=FilterStepAction,
        # rename=RenameAction,
        # remapping=RemappingAction,
    ).get(key)

    if isinstance(config[key], list):
        args, kwargs = config[key], {}

    if isinstance(config[key], dict):
        args, kwargs = [], config[key]

    if isinstance(config[key], str):
        args, kwargs = [config[key]], {}

    if cls is not None:
        return cls(context, action_path, previous_step, *args, **kwargs)

    # Try filters from datasets filter registry
    from anemoi.transform.filters import filter_registry as transform_filter_registry

    from ..filters import create_filter as create_datasets_filter
    from ..filters import filter_registry as datasets_filter_registry

    if datasets_filter_registry.is_registered(key):

        if transform_filter_registry.is_registered(key):
            warnings.warn(f"Filter `{key}` is registered in both datasets and transform filter registries")

        filter = create_datasets_filter(None, config)
        return FunctionStepAction(context, action_path + [key], previous_step, key, filter)

    # Use filters from transform registry

    if transform_filter_registry.is_registered(key):
        from ..filters.transform import TransformFilter

        return FunctionStepAction(
            context, action_path + [key], previous_step, key, TransformFilter(context, key, config)
        )

    raise ValueError(f"Unknown step action `{key}`")
