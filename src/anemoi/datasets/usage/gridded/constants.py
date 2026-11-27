# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.utils.dates import to_datetime
from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.dataset import Shape
from anemoi.datasets.usage.dataset import TupleIndex
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.debug import debug_indexing
from anemoi.datasets.usage.forwards import Forwards
from anemoi.datasets.usage.gridded.indexing import apply_index_to_slices_changes
from anemoi.datasets.usage.gridded.indexing import expand_list_indexing
from anemoi.datasets.usage.gridded.indexing import index_to_slices
from anemoi.datasets.usage.gridded.indexing import update_tuple
from anemoi.datasets.usage.misc import _open_dataset
from anemoi.datasets.usage.options import Options

LOG = logging.getLogger(__name__)


class Constants(Forwards):
    """Class to handle constant values in a dataset."""

    def __init__(
        self,
        dataset: Dataset,
        start: datetime.datetime,
        end: datetime.datetime,
        frequency: datetime.timedelta,
        caching: bool = False,
    ) -> None:
        """Initialize the Constants class.

        Parameters
        ----------
        dataset : Dataset
            The dataset with constant values.
        start : datetime
            The start date for the constants.
        end : datetime
            The end date for the constants.
        frequency : timedelta
            The frequency of the constants.
        caching : bool, optional
            Whether to enable caching, by default False.
        """
        super().__init__(dataset)

        self._start = start
        self._end = end
        self._frequency = frequency
        self._caching = caching
        self._cache = None
        self._index = 0  # Index of the constant values in the source dataset

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get a tuple of data for the given index.

        Parameters
        ----------
        index : TupleIndex
            The index to retrieve data for.

        Returns
        -------
        NDArray[Any]
            The data for the given index.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 0, slice(None))
        result = self._get_slice(previous)
        return apply_index_to_slices_changes(result[index], changes)

    def _get_slice(self, s: slice) -> NDArray[Any]:
        """Get a slice of data.

        Parameters
        ----------
        s : slice
            The slice to retrieve data for.

        Returns
        -------
        NDArray[Any]
            The data for the given slice.
        """
        return np.stack([self[i] for i in range(*s.indices(self._len))])

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def frequency(self) -> datetime.timedelta:
        """Return the frequency of the dataset."""
        return self._frequency

    @property
    def dates(self) -> NDArray[np.datetime64]:
        return np.arange(self._start, self._end, self._frequency, dtype="datetime64")

    @property
    def shape(self) -> Shape:
        return (len(self.dates), self.forward.shape[1:])

    @property
    def missing(self) -> set[int]:
        """Get the set of missing dates."""
        return set()

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get the data for the given index.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve data for.

        Returns
        -------
        NDArray[Any]
            The data for the given index.
        """

        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        assert isinstance(n, int), n

        if self._caching and self._cache is not None:
            return self._cache

        result = self.forward[self._index]
        if self._caching:
            self._cache = result
        return result

    def forwards_subclass_metadata_specific(self) -> dict:
        """Return subclass-specific metadata for the forward dataset."""
        return dict(
            start=str(self._start),
            end=str(self._end),
            frequency=frequency_to_string(self._frequency),
        )

    def tree(self) -> Node:
        """Get the tree representation of the object.

        Returns
        -------
        Node
            The tree representation of the object.
        """
        return Node(self, [self.forward.tree()], start=self._start, end=self._end, frequency=self._frequency)


def constants_factory(args: tuple, kwargs: dict, options: Options) -> Dataset:
    """Factory function to create a Complement instance based on the provided arguments.

    Parameters
    ----------
    args : Tuple
        Positional arguments.
    kwargs : dict
        Keyword arguments.
    options : Options
        The options to use when opening the datasets.

    Returns
    -------
    Dataset
        The complemented dataset.
    """

    assert len(args) == 0, args

    constants = kwargs.pop("constants")

    # First, consume start, end, frequency, and caching from kwargs
    start = constants.pop("start", None)
    end = constants.pop("end", None)
    frequency = constants.pop("frequency", None)
    caching = constants.pop("caching", False)

    # Open the dataset with the rest

    source = _open_dataset(constants, options=options)

    start = to_datetime(start or source.start_date)
    end = to_datetime(end or source.end_date)
    frequency = frequency_to_timedelta(frequency or source.frequency)

    return Constants(source, start=start, end=end, frequency=frequency, caching=caching)
