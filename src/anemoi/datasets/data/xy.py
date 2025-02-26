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
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

from .dataset import Dataset
from .dataset import FullIndex
from .debug import Node
from .forwards import Combined
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class ZipBase(Combined):
    """Base class for handling zipped datasets."""

    def __init__(self, datasets: List[Any], check_compatibility: bool = True) -> None:
        """Initialize ZipBase with a list of datasets.

        Parameters
        ----------
        datasets : List[Any]
            List of datasets.
        check_compatibility : bool, optional
            Flag to check compatibility of datasets, by default True.
        """
        self._check_compatibility = check_compatibility
        super().__init__(datasets)

    def swap_with_parent(self, parent: Any) -> Any:
        """Swap datasets with the parent.

        Parameters
        ----------
        parent : Any
            Parent dataset.

        Returns
        -------
        Any
            New parent dataset with swapped datasets.
        """
        new_parents = [parent.clone(ds) for ds in self.datasets]
        return self.clone(new_parents)

    def clone(self, datasets: List[Any]) -> "ZipBase":
        """Clone the ZipBase with new datasets.

        Parameters
        ----------
        datasets : List[Any]
            List of new datasets.

        Returns
        -------
        ZipBase
            Cloned ZipBase instance.
        """
        return self.__class__(datasets, check_compatibility=self._check_compatibility)

    def tree(self) -> Node:
        """Get the tree representation of the datasets.

        Returns
        -------
        Node
            Tree representation of the datasets.
        """
        return Node(self, [d.tree() for d in self.datasets], check_compatibility=self._check_compatibility)

    def __len__(self) -> int:
        """Get the length of the smallest dataset.

        Returns
        -------
        int
            Length of the smallest dataset.
        """
        return min(len(d) for d in self.datasets)

    def __getitem__(self, n: FullIndex) -> Tuple[Any, ...]:
        """Get the item at the specified index from all datasets.

        Parameters
        ----------
        n : FullIndex
            Index to retrieve.

        Returns
        -------
        Tuple[Any, ...]
            Tuple of items from all datasets.
        """
        return tuple(d[n] for d in self.datasets)

    def check_same_resolution(self, d1: Dataset, d2: Dataset) -> None:
        """Check if two datasets have the same resolution.

        Parameters
        ----------
        d1 : Dataset
            First dataset.
        d2 : Dataset
            Second dataset.
        """
        pass

    def check_same_grid(self, d1: Dataset, d2: Dataset) -> None:
        """Check if two datasets have the same grid.

        Parameters
        ----------
        d1 : Dataset
            First dataset.
        d2 : Dataset
            Second dataset.
        """
        pass

    def check_same_variables(self, d1: Dataset, d2: Dataset) -> None:
        """Check if two datasets have the same variables.

        Parameters
        ----------
        d1 : Dataset
            First dataset.
        d2 : Dataset
            Second dataset.
        """
        pass

    @cached_property
    def missing(self) -> Set[int]:
        """Get the set of missing indices from all datasets."""
        result: Set[int] = set()
        for d in self.datasets:
            result = result | d.missing
        return result

    @property
    def shape(self) -> Tuple[Any, ...]:
        """Get the shape of all datasets."""
        return tuple(d.shape for d in self.datasets)

    @property
    def field_shape(self) -> Tuple[Any, ...]:
        """Get the field shape of all datasets."""
        return tuple(d.shape for d in self.datasets)

    @property
    def latitudes(self) -> Tuple[Any, ...]:
        """Get the latitudes of all datasets."""
        return tuple(d.latitudes for d in self.datasets)

    @property
    def longitudes(self) -> Tuple[Any, ...]:
        """Get the longitudes of all datasets."""
        return tuple(d.longitudes for d in self.datasets)

    @property
    def dtype(self) -> Tuple[Any, ...]:
        """Get the data types of all datasets."""
        return tuple(d.dtype for d in self.datasets)

    @property
    def grids(self) -> Tuple[Any, ...]:
        """Get the grids of all datasets."""
        return tuple(d.grids for d in self.datasets)

    @property
    def statistics(self) -> Tuple[Any, ...]:
        """Get the statistics of all datasets."""
        return tuple(d.statistics for d in self.datasets)

    @property
    def resolution(self) -> Tuple[Any, ...]:
        """Get the resolution of all datasets."""
        return tuple(d.resolution for d in self.datasets)

    @property
    def name_to_index(self) -> Tuple[Any, ...]:
        """Get the name to index mapping of all datasets."""
        return tuple(d.name_to_index for d in self.datasets)

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Check compatibility between two datasets.

        Parameters
        ----------
        d1 : Dataset
            First dataset.
        d2 : Dataset
            Second dataset.
        """
        if self._check_compatibility:
            super().check_compatibility(d1, d2)


class Zip(ZipBase):
    """Class for handling zipped datasets."""

    pass


class XY(ZipBase):
    """Class for handling XY datasets."""

    pass


def xy_factory(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> XY:
    """Factory function to create an XY instance.

    Parameters
    ----------
    args : Tuple[Any, ...]
        Positional arguments.
    kwargs : Dict[str, Any]
        Keyword arguments.

    Returns
    -------
    XY
        An instance of XY.
    """
    if "xy" in kwargs:
        xy = kwargs.pop("xy")
    else:
        xy = [kwargs.pop("x"), kwargs.pop("y")]

    assert len(args) == 0
    assert isinstance(xy, (list, tuple))

    datasets = [_open(e) for e in xy]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    assert len(datasets) == 2

    check_compatibility = kwargs.pop("check_compatibility", True)

    return XY(datasets, check_compatibility=check_compatibility)._subset(**kwargs)


def zip_factory(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Zip:
    """Factory function to create a Zip instance.

    Parameters
    ----------
    args : Tuple[Any, ...]
        Positional arguments.
    kwargs : Dict[str, Any]
        Keyword arguments.

    Returns
    -------
    Zip
        An instance of Zip.
    """
    zip = kwargs.pop("zip")
    assert len(args) == 0
    assert isinstance(zip, (list, tuple))

    datasets = [_open(e) for e in zip]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    check_compatibility = kwargs.pop("check_compatibility", True)

    return Zip(datasets, check_compatibility=check_compatibility)._subset(**kwargs)
