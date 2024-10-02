# (C) Copyright 2023 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import datetime
import itertools
import logging
import math
import time
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from functools import wraps

import numpy as np
from anemoi.utils.dates import as_datetime as as_datetime
from anemoi.utils.dates import frequency_to_timedelta as frequency_to_timedelta

from anemoi.datasets.dates import DatesProvider as DatesProvider
from anemoi.datasets.fields import FieldArray as FieldArray
from anemoi.datasets.fields import NewValidDateTimeField as NewValidDateTimeField

from .trace import trace_select

LOG = logging.getLogger(__name__)


class InputBuilder:
    def __init__(self, config, data_sources, **kwargs):
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
    def select(self, group_of_dates):
        from .action import ActionContext
        from .action import action_factory

        """This changes the context."""
        context = ActionContext(**self.kwargs)
        action = action_factory(self.config, context, self.action_path)
        return action.select(group_of_dates)

    def __repr__(self):
        from .action import ActionContext
        from .action import action_factory

        context = ActionContext(**self.kwargs)
        a = action_factory(self.config, context, self.action_path)
        return repr(a)

    def _trace_select(self, group_of_dates):
        return f"InputBuilder({group_of_dates})"


build_input = InputBuilder
