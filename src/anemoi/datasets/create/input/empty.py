# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

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
