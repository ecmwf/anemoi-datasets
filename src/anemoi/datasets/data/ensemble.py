# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Dict
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .debug import Node
from .forwards import Forwards
from .forwards import GivenAxis
from .indexing import apply_index_to_slices_changes
from .indexing import index_to_slices
from .indexing import update_tuple
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)

OFFSETS = dict(number=1, numbers=1, member=0, members=0)


class Number(Forwards):
    """A class to represent a subset of ensemble members from a dataset."""

    def __init__(self, forward: Dataset, **kwargs: Any) -> None:
        """Initializes a Number object.

        Parameters
        ----------
        forward : Any
            The dataset to forward.
        kwargs : Any
            Additional keyword arguments specifying the members.
        """
        super().__init__(forward)

        self.members = []
        for key, values in kwargs.items():
            if not isinstance(values, (list, tuple)):
                values = [values]
            self.members.extend([int(v) - OFFSETS[key] for v in values])

        self.members = sorted(set(self.members))
        for n in self.members:
            if not (0 <= n < forward.shape[2]):
                raise ValueError(f"Member {n} is out of range. `number(s)` is one-based, `member(s)` is zero-based.")

        self.mask = np.array([n in self.members for n in range(forward.shape[2])], dtype=bool)
        self._shape, _ = update_tuple(forward.shape, 2, len(self.members))

    @property
    def shape(self) -> Shape:
        """Returns the shape of the dataset."""
        return self._shape

    def __getitem__(self, index: FullIndex) -> NDArray[Any]:
        """Retrieves data from the dataset based on the given index.

        Parameters
        ----------
        index : FullIndex
            Index specifying the data to retrieve.

        Returns
        -------
        NDArray[Any]
            Data array from the dataset based on the index.
        """
        if isinstance(index, int):
            result = self.forward[index]
            result = result[:, self.mask, :]
            return result

        if isinstance(index, slice):
            result = self.forward[index]
            result = result[:, :, self.mask, :]
            return result

        index, changes = index_to_slices(index, self.forward.shape)
        result = self.forward[index]
        result = result[:, :, self.mask, :]
        return apply_index_to_slices_changes(result, changes)

    def tree(self) -> Node:
        """Generates a hierarchical tree structure for the dataset.

        Returns
        -------
        Node
            A Node object representing the dataset.
        """
        return Node(self, [self.forward.tree()], numbers=[n + 1 for n in self.members])

    def metadata_specific(self, **kwargs: Any) -> Dict[str, Any]:
        """Returns metadata specific to the Number object.

        Parameters
        ----------
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Dict[str, Any]
            Metadata specific to the Number object.
        """
        return {
            "numbers": [n + 1 for n in self.members],
        }

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Returns metadata specific to the Number object."""
        return {}


class Ensemble(GivenAxis):
    """A class to represent an ensemble of datasets combined along a given axis."""

    def tree(self) -> Node:
        """Generates a hierarchical tree structure for the ensemble datasets.

        Returns
        -------
        Node
            A Node object representing the ensemble datasets.
        """
        return Node(self, [d.tree() for d in self.datasets])

    def forwards_subclass_metadata_specific(self) -> Dict[str, Any]:
        """Get the metadata specific to the forwards subclass.

        Returns:
        Dict[str, Any]: The metadata specific to the forwards subclass.
        """
        return {}


def ensemble_factory(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Ensemble:
    """Factory function to create an Ensemble object.

    Parameters
    ----------
    args : Tuple[Any, ...]
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    Ensemble
        An Ensemble object.
    """
    if "grids" in kwargs:
        raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")

    ensemble = kwargs.pop("ensemble")
    axis = kwargs.pop("axis", 2)
    assert len(args) == 0
    assert isinstance(ensemble, (list, tuple))

    datasets = [_open(e) for e in ensemble]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Ensemble(datasets, axis=axis)._subset(**kwargs)
