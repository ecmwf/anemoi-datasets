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

from anemoi.utils.dates import as_datetime as as_datetime
from anemoi.utils.dates import frequency_to_timedelta as frequency_to_timedelta

from anemoi.datasets.dates import DatesProvider as DatesProvider
from anemoi.datasets.fields import FieldArray as FieldArray
from anemoi.datasets.fields import NewValidDateTimeField as NewValidDateTimeField

from ..functions import import_function
from .function import FunctionContext
from .misc import _tidy
from .misc import assert_fieldlist
from .step import StepAction
from .step import StepResult
from .template import notify_result
from .trace import trace_datasource

LOG = logging.getLogger(__name__)


class FilterStepResult(StepResult):
    @property
    @notify_result
    @assert_fieldlist
    @trace_datasource
    def datasource(self):
        ds = self.upstream_result.datasource
        ds = ds.sel(**self.action.kwargs)
        return _tidy(ds)


class FilterStepAction(StepAction):
    result_class = FilterStepResult


class StepFunctionResult(StepResult):
    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self):
        try:
            return _tidy(
                self.action.function(
                    FunctionContext(self),
                    self.upstream_result.datasource,
                    *self.action.args[1:],
                    **self.action.kwargs,
                )
            )

        except Exception:
            LOG.error(f"Error in {self.action.name}", exc_info=True)
            raise

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.action.name}({self.group_of_dates})"


class FunctionStepAction(StepAction):
    result_class = StepFunctionResult

    def __init__(self, context, action_path, previous_step, *args, **kwargs):
        super().__init__(context, action_path, previous_step, *args, **kwargs)
        self.name = args[0]
        self.function = import_function(self.name, "filters")
