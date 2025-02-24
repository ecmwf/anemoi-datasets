# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set

from numpy.typing import NDArray

from . import open_dataset
from .dataset import Dataset
from .debug import Node
from .forwards import Forwards

LOG = logging.getLogger(__name__)


class Statistics(Forwards):
    """A class to represent statistics for a dataset.

    Attributes
    ----------
    dataset : Dataset
        The dataset object.
    statistic : Any
        The statistic data.
    """

    def __init__(self, dataset: Dataset, statistic: Any) -> None:
        """Initialize the Statistics object.

        Parameters
        ----------
        dataset : Dataset
            The dataset object.
        statistic : Any
            The statistic data.
        """
        super().__init__(dataset)
        self._statistic = open_dataset(statistic, select=dataset.variables)
        # TODO: relax that check to allow for a subset of variables
        if dataset.variables != self._statistic.variables:
            raise ValueError(
                f"Incompatible variables: {dataset.variables} and {self._statistic.variables} ({dataset} {self._statistic})"
            )

    @cached_property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        """Get the statistics."""
        return self._statistic.statistics

    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        """Get the statistics tendencies.

        Parameters
        ----------
        delta : Optional[datetime.timedelta]
            The time delta.

        Returns
        -------
        Dict[str, NDArray[Any]]
            The statistics tendencies.
        """
        if delta is None:
            delta = self.frequency
        return self._statistic.statistics_tendencies(delta)

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the forwards subclass.
        """
        return dict(statistics=self._statistic.metadata_specific())

    def tree(self) -> Node:
        """Get the tree representation of the statistics.

        Returns
        -------
        Node
            The tree representation of the statistics.
        """
        return Node(self, [self.forward.tree()])

    def get_dataset_names(self, names: Set[str]) -> None:
        """Get the dataset names.

        Parameters
        ----------
        names : Set[str]
            The set of dataset names.
        """
        super().get_dataset_names(names)
        self._statistic.get_dataset_names(names)
