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
from .forwards import Forwards

LOG = logging.getLogger(__name__)


class Statistics(Forwards):
    def __init__(self, dataset, statistic):
        super().__init__(dataset)
        self._statistic = open_dataset(statistic, select=dataset.variables)
        # TODO: relax that check to allow for a subset of variables
        if dataset.variables != self._statistic.variables:
            raise ValueError(
                f"Incompatible variables: {dataset.variables} and {self._statistic.variables} ({dataset} {self._statistic})"
            )

    @cached_property
    def statistics(self):
        return self._statistic.statistics

    def statistics_tendencies(self, delta=None):
        if delta is None:
            delta = self.frequency
        return self._statistic.statistics_tendencies(delta)

    def subclass_metadata_specific(self):
        return dict(statistics=self._statistic.metadata_specific())

    def tree(self):
        return Node(self, [self.forward.tree()])

    def get_dataset_names(self, names):
        super().get_dataset_names(names)
        self._statistic.get_dataset_names(names)
