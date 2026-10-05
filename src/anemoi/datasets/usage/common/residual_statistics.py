# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ``residual_statistics`` option of ``open_dataset``.

Warnings
--------

Experimental: this option and the ``residual_statistics`` attribute it
provides may be removed or renamed in a future release.

Attaches the statistics of the difference between two datasets, read from a JSON
file written by ``anemoi-datasets compute ... --minus ...``, to a
dataset. The file format is described in
:mod:`anemoi.datasets.misc.residual_statistics`.

This wrapper lives in ``common`` because it is layout-agnostic: residual
statistics are indexed by variable only, exactly like ``statistics``, so the
gridded, tabular and trajectory layouts can all use it.
"""

import logging
import os
from functools import cached_property
from typing import Any

from numpy.typing import NDArray

from anemoi.datasets.misc.residual_statistics import ResidualStatisticsFile
from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.forwards import Forwards

LOG = logging.getLogger(__name__)


class ResidualStatistics(Forwards):
    """A dataset with residual statistics attached from a JSON file.

    Attributes
    ----------
    file : ResidualStatisticsFile
        The validated residual-statistics file.
    """

    def __init__(
        self, dataset: Dataset, residual_statistics: "str | os.PathLike[str] | ResidualStatisticsFile"
    ) -> None:
        """Initialize the ResidualStatistics object.

        Parameters
        ----------
        dataset : Dataset
            The forward dataset.
        residual_statistics : str or path-like or ResidualStatisticsFile
            Path to the residual-statistics JSON file, or an already loaded
            file. The file must cover every variable of ``dataset``; extra
            variables in the file are ignored.
        """
        super().__init__(dataset)

        if isinstance(residual_statistics, ResidualStatisticsFile):
            self.file = residual_statistics
        else:
            self.file = ResidualStatisticsFile.load(residual_statistics)

        # Fail at open time rather than on first access to `residual_statistics`.
        self.file.select(dataset.variables)

    @cached_property
    def residual_statistics(self) -> dict[str, NDArray[Any]]:
        """Get the residual statistics, indexed like :attr:`variables`.

        Experimental: may be removed or renamed in a future release.
        """
        return self.file.select(self.variables)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the forwards subclass.
        """
        return {"residual_statistics": self.file.metadata()}

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation.
        """
        return Node(self, [self.forward.tree()], residual_statistics=self.file.path)

    def __repr__(self) -> str:
        """Return the string representation of the dataset."""
        return f"{self.__class__.__name__}({self.forward}, {self.file.path})"
