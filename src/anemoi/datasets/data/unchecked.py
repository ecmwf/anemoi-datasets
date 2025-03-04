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
from functools import wraps
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy as np
from numpy.typing import NDArray

from .concat import ConcatMixin
from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .debug import Node
from .forwards import Combined
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class check:
    """A decorator class to perform checks before calling a method."""

    def __init__(self, check: str) -> None:
        """Initialize the check decorator.

        Parameters
        ----------
        check : str
            The name of the check method.
        """
        self.check = check

    def __call__(self, method: Callable) -> Callable:
        """Call the check decorator.

        Parameters
        ----------
        method : Callable
            The method to decorate.

        Returns
        -------
        Callable
            The decorated method.
        """
        name = method.__name__
        check = self.check

        @wraps(method)
        def wrapper(obj: "Unchecked") -> Any:
            """Wrapper function to check compatibility before calling the method.

            Parameters
            ----------
            obj : Unchecked
                The Unchecked object.

            Returns
            -------
            Any
                The result of the method.
            """
            for d in obj.datasets[1:]:
                getattr(obj, check)(obj.datasets[0], d)

            return getattr(Combined, name).__get__(obj)

        return wrapper


class Unchecked(Combined):
    """A class representing a dataset without compatibility checks."""

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation.
        """
        return Node(self, [d.tree() for d in self.datasets])

    def _subset(self, **kwargs: dict) -> "Unchecked":
        """Get a subset of the dataset.

        Parameters
        ----------
        **kwargs : dict
            Subset parameters.

        Returns
        -------
        Unchecked
            The subset of the dataset.
        """
        assert not kwargs
        return self

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Check compatibility between two datasets.

        Parameters
        ----------
        d1 : Dataset
            The first dataset.
        d2 : Dataset
            The second dataset.
        """
        pass

    @property
    @check("check_same_dates")
    def dates(self) -> NDArray[np.datetime64]:
        """Get the dates of the dataset."""
        pass

    @property
    @check("check_same_resolution")
    def resolution(self) -> Any:
        """Get the resolution of the dataset."""
        pass

    @property
    def field_shape(self) -> tuple:
        """Get the field shape of the dataset."""
        raise NotImplementedError()

    @property
    @check("check_same_frequency")
    def frequency(self) -> datetime.timedelta:
        """Get the frequency of the dataset."""
        raise NotImplementedError()

    @property
    @check("check_same_grid")
    def latitudes(self) -> NDArray[Any]:
        """Get the latitudes of the dataset."""
        raise NotImplementedError()

    @property
    @check("check_same_grid")
    def longitudes(self) -> NDArray[Any]:
        """Get the longitudes of the dataset."""
        raise NotImplementedError()

    @check("check_same_variables")
    @property
    def name_to_index(self) -> Dict[str, int]:
        """Get the mapping of variable names to their indices."""
        raise NotImplementedError()

    @check("check_same_variables")
    @property
    def variables(self) -> List[str]:
        """Get the list of variables in the dataset."""
        raise NotImplementedError()

    @check("check_same_variables")
    @property
    def variables_metadata(self) -> dict:
        """Get the metadata for the variables."""
        raise NotImplementedError()

    @check("check_same_variables")
    @property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        """Get the statistics of the dataset."""
        raise NotImplementedError()

    @check("check_same_variables")
    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        """Get the statistics tendencies of the dataset.

        Parameters
        ----------
        delta : Optional[datetime.timedelta]
            The time delta for tendencies.

        Returns
        -------
        Dict[str, NDArray[Any]]
            The statistics tendencies.
        """
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        """Get the shape of the dataset."""
        raise NotImplementedError()

    @cached_property
    def missing(self) -> Set[int]:
        """Get the missing data indices."""
        result: Set[int] = set()
        for d in self.datasets:
            result = result | d.missing
        return result


class Chain(ConcatMixin, Unchecked):
    """A class representing a chain of datasets without compatibility checks."""

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, n: FullIndex) -> tuple:
        """Get an item from the dataset.

        Parameters
        ----------
        n : FullIndex
            The index of the item.

        Returns
        -------
        tuple
            The item at the specified index.
        """
        return tuple(d[n] for d in self.datasets)

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Get the dates of the dataset."""
        raise NotImplementedError()

    def dataset_metadata(self) -> dict:
        """Get the metadata of the dataset.

        Returns
        -------
        dict
            The metadata of the dataset.
        """
        return {"multiple": [d.dataset_metadata() for d in self.datasets]}


def chain_factory(args: tuple, kwargs: dict) -> Dataset:
    """Factory function to create a Chain dataset.

    Parameters
    ----------
    args : tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.

    Returns
    -------
    Dataset
        The Chain dataset.
    """
    chain = kwargs.pop("chain")
    assert len(args) == 0
    assert isinstance(chain, (list, tuple))

    datasets = [_open(e) for e in chain]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Chain(datasets)._subset(**kwargs)
