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

from .misc import assert_fieldlist
from .result import Result
from .trace import trace_datasource

LOG = logging.getLogger(__name__)


class EmptyResult(Result):
    empty = True

    def __init__(self, context, action_path, dates):
        super().__init__(context, action_path + ["empty"], dates)

    @cached_property
    @assert_fieldlist
    @trace_datasource
    def datasource(self):
        from earthkit.data import from_source

        return from_source("empty")

    @property
    def variables(self):
        return []
