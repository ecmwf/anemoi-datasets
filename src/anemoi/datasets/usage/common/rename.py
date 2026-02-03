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
from typing import Any

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.forwards import Forwards

LOG = logging.getLogger(__name__)


class Rename(Forwards):
    """Class to rename variables in a dataset."""

    def __init__(self, dataset: Dataset, rename: dict[str, str]) -> None:
        """Initialize the Rename class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to rename.
        rename : Dict[str, str]
            The mapping of old names to new names.
        """
        super().__init__(dataset)
        for n in rename:
            assert n in dataset.variables, n

        self._variables = [rename.get(v, v) for v in dataset.variables]
        self._variables_metadata = {rename.get(k, k): v for k, v in dataset.variables_metadata.items()}

        self.rename = rename

    @property
    def variables(self) -> list[str]:
        """Get the renamed variables."""
        return self._variables

    @property
    def variables_metadata(self) -> dict[str, Any]:
        """Get the renamed variables metadata."""
        return self._variables_metadata

    @cached_property
    def name_to_index(self) -> dict[str, int]:
        """Get the mapping of renamed variable names to indices."""
        return {k: i for i, k in enumerate(self.variables)}

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns:
            Node: The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], rename=self.rename)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the subclass.

        Returns:
            Dict[str, Any]: The metadata specific to the subclass.
        """
        return dict(rename=self.rename)

    def origin_transformation(self, variable, origins):
        return {
            "name": "rename",
            "config": {"rename": self.rename},
        }, self.rename.get(variable, variable)
