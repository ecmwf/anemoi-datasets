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

from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.dataset import TupleIndex
from anemoi.datasets.usage.debug import debug_indexing
from anemoi.datasets.usage.gridded.indexing import apply_index_to_slices_changes
from anemoi.datasets.usage.gridded.indexing import expand_list_indexing
from anemoi.datasets.usage.gridded.indexing import index_to_slices
from anemoi.datasets.usage.gridded.indexing import update_tuple

from ..common.select import SelectBase

LOG = logging.getLogger(__name__)


class Select(SelectBase):
    """Class to select a subset of variables from a dataset."""

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index: TupleIndex) -> NDArray[Any]:
        """Get a tuple of data.

        Parameters
        ----------
        index : TupleIndex
            The index to retrieve.

        Returns
        -------
        NDArray[Any]
            The retrieved data.
        """
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 1, slice(None))
        result = self.dataset[index]
        result = result[:, self.indices]
        result = result[:, previous]
        result = apply_index_to_slices_changes(result, changes)
        return result

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get an item from the dataset.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve.

        Returns
        -------
        NDArray[Any]
            The retrieved data.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        row = self.dataset[n]
        if isinstance(n, slice):
            return row[:, self.indices]

        return row[self.indices]
