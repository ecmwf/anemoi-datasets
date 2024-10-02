# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from copy import deepcopy

from anemoi.utils.dates import as_datetime as as_datetime
from anemoi.utils.dates import frequency_to_timedelta as frequency_to_timedelta

from anemoi.datasets.dates import DatesProvider as DatesProvider
from anemoi.datasets.fields import FieldArray as FieldArray
from anemoi.datasets.fields import NewValidDateTimeField as NewValidDateTimeField

from .action import Action
from .context import Context
from .misc import is_function
from .result import Result
from .template import notify_result
from .trace import trace_datasource
from .trace import trace_select

LOG = logging.getLogger(__name__)


class StepResult(Result):
    def __init__(self, context, action_path, group_of_dates, action, upstream_result):
        super().__init__(context, action_path, group_of_dates)
        assert isinstance(upstream_result, Result), type(upstream_result)
        self.upstream_result = upstream_result
        self.action = action

    @property
    @notify_result
    @trace_datasource
    def datasource(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")


class StepAction(Action):
    result_class = None

    def __init__(self, context, action_path, previous_step, *args, **kwargs):
        super().__init__(context, action_path, *args, **kwargs)
        self.previous_step = previous_step

    @trace_select
    def select(self, group_of_dates):
        return self.result_class(
            self.context,
            self.action_path,
            group_of_dates,
            self,
            self.previous_step.select(group_of_dates),
        )

    def __repr__(self):
        return super().__repr__(self.previous_step, _inline_=str(self.kwargs))


def step_factory(config, context, action_path, previous_step):

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

    if cls is None:
        if not is_function(key, "filters"):
            raise ValueError(f"Unknown step {key}")
        cls = FunctionStepAction
        args = [key] + args

    return cls(context, action_path, previous_step, *args, **kwargs)
