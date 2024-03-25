# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

from . import open_dataset
from .debug import Node
from .forewards import Forwards

LOG = logging.getLogger(__name__)


class Statistics(Forwards):
    def __init__(self, dataset, statistic):
        super().__init__(dataset)
        self._statistic = statistic

    @cached_property
    def statistics(self):
        return open_dataset(self._statistic).statistics

    def metadata_specific(self, **kwargs):
        return super().metadata_specific(
            statistics=open_dataset(self._statistic).metadata_specific(),
            **kwargs,
        )

    def tree(self):
        return Node(self, [self.forward.tree()])
