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

from numpy.typing import NDArray

from anemoi.datasets import open_dataset
from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.forwards import Forwards

LOG = logging.getLogger(__name__)


class Statistics(Forwards):
    """A class to represent statistics for a dataset.

    Attributes
    ----------
    dataset : Dataset
        The dataset object.
    statistic : Any
        Dataset providing the override for ``statistics``. Optional.
    statistic_tendencies : Any
        Dataset providing the override for ``statistics_tendencies``. Optional.
    """

    def __init__(
        self,
        dataset: Dataset,
        statistic: Any = None,
        statistic_tendencies: Any = None,
    ) -> None:
        """Initialize the Statistics object.

        Parameters
        ----------
        dataset : Dataset
            The forward dataset.
        statistic : Any, optional
            Dataset whose ``statistics`` will be used as override. If ``None``,
            ``statistics`` falls back to the forward dataset.
        statistic_tendencies : Any, optional
            Dataset whose ``statistics_tendencies`` will be used as override. If
            ``None``, tendencies fall back to ``statistic`` (when provided) or to
            the forward dataset otherwise.
        """
        super().__init__(dataset)
        if statistic is None and statistic_tendencies is None:
            raise ValueError(
                "Statistics requires at least one of `statistic` or `statistic_tendencies` to be provided."
            )
        self._statistic = self._open_compatible(dataset, statistic)
        self._statistic_tendencies = self._open_compatible(dataset, statistic_tendencies)

    @staticmethod
    def _open_compatible(dataset: Dataset, source: Any) -> Any:
        """Open ``source`` and check that its variables match ``dataset``."""
        if source is None:
            return None
        ds = open_dataset(source, select=dataset.variables)
        # TODO: relax that check to allow for a subset of variables
        if dataset.variables != ds.variables:
            raise ValueError(f"Incompatible variables: {dataset.variables} and {ds.variables} ({dataset} {ds})")
        return ds

    @cached_property
    def statistics(self) -> dict[str, NDArray[Any]]:
        """Get the statistics."""
        if self._statistic is not None:
            return self._statistic.statistics
        return self.forward.statistics

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
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
        if self._statistic_tendencies is not None:
            return self._statistic_tendencies.statistics_tendencies(delta)
        if self._statistic is not None:
            return self._statistic.statistics_tendencies(delta)
        return self.forward.statistics_tendencies(delta)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the forwards subclass.
        """
        md: dict[str, Any] = {}
        if self._statistic is not None:
            md["statistics"] = self._statistic.metadata_specific()
        if self._statistic_tendencies is not None:
            md["statistics_tendencies"] = self._statistic_tendencies.metadata_specific()
        return md

    def tree(self) -> Node:
        """Get the tree representation of the statistics.

        Returns
        -------
        Node
            The tree representation of the statistics.
        """
        return Node(self, [self.forward.tree()])

    def get_dataset_names(self, names: set[str]) -> None:
        """Get the dataset names.

        Parameters
        ----------
        names : Set[str]
            The set of dataset names.
        """
        super().get_dataset_names(names)
        if self._statistic is not None:
            self._statistic.get_dataset_names(names)
        if self._statistic_tendencies is not None:
            self._statistic_tendencies.get_dataset_names(names)
