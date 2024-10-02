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
from .action import Action
from .misc import _tidy
from .misc import assert_fieldlist
from .result import Result
from .template import notify_result
from .template import resolve
from .template import substitute
from .trace import trace
from .trace import trace_datasource
from .trace import trace_select

LOG = logging.getLogger(__name__)


class FunctionContext:
    """A FunctionContext is passed to all functions, it will be used to pass information
    to the functions from the other actions and filters and results.
    """

    def __init__(self, owner):
        self.owner = owner
        self.use_grib_paramid = owner.context.use_grib_paramid

    def trace(self, emoji, *args):
        trace(emoji, *args)

    def info(self, *args, **kwargs):
        LOG.info(*args, **kwargs)

    @property
    def dates_provider(self):
        return self.owner.group_of_dates.provider

    @property
    def partial_ok(self):
        return self.owner.group_of_dates.partial_ok


class FunctionAction(Action):
    def __init__(self, context, action_path, _name, **kwargs):
        super().__init__(context, action_path, **kwargs)
        self.name = _name

    @trace_select
    def select(self, group_of_dates):
        return FunctionResult(self.context, self.action_path, group_of_dates, action=self)

    @property
    def function(self):
        # name, delta = parse_function_name(self.name)
        return import_function(self.name, "sources")

    def __repr__(self):
        content = ""
        content += ",".join([self._short_str(a) for a in self.args])
        content += " ".join([self._short_str(f"{k}={v}") for k, v in self.kwargs.items()])
        content = self._short_str(content)
        return super().__repr__(_inline_=content, _indent_=" ")

    def _trace_select(self, group_of_dates):
        return f"{self.name}({group_of_dates})"


class FunctionResult(Result):
    def __init__(self, context, action_path, group_of_dates, action):
        super().__init__(context, action_path, group_of_dates)
        assert isinstance(action, Action), type(action)
        self.action = action

        self.args, self.kwargs = substitute(context, (self.action.args, self.action.kwargs))

    def _trace_datasource(self, *args, **kwargs):
        return f"{self.action.name}({self.group_of_dates})"

    @cached_property
    @assert_fieldlist
    @notify_result
    @trace_datasource
    def datasource(self):
        args, kwargs = resolve(self.context, (self.args, self.kwargs))

        try:
            return _tidy(
                self.action.function(
                    FunctionContext(self),
                    list(self.group_of_dates),  # Will provide a list of datetime objects
                    *args,
                    **kwargs,
                )
            )
        except Exception:
            LOG.error(f"Error in {self.action.function.__name__}", exc_info=True)
            raise

    def __repr__(self):
        try:
            return f"{self.action.name}({self.group_of_dates})"
        except Exception:
            return f"{self.__class__.__name__}(unitialised)"

    @property
    def function(self):
        raise NotImplementedError(f"Not implemented in {self.__class__.__name__}")
