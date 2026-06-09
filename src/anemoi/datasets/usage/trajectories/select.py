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
from anemoi.datasets.usage.debug import debug_indexing

from ..common.select import SelectBase

LOG = logging.getLogger(__name__)


class Select(SelectBase):
    """Select a subset of variables from a trajectories dataset.

    The variable axis is at position 1 in the 5-D layout
    ``(dates, variables, ensembles, steps, cells)``.
    """

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get data with variable subsetting applied.

        Parameters
        ----------
        n : FullIndex
            Index into the date axis (int, slice, or tuple).

        Returns
        -------
        NDArray[Any]
            The data with only the selected variables.
        """
        if isinstance(n, tuple):
            return self._get_tuple(n)

        row = self.dataset[n]
        if isinstance(n, slice):
            # (dates, variables, ensembles, steps, cells)
            return row[:, self.indices]

        # scalar n → (variables, ensembles, steps, cells)
        return row[self.indices]

    def _get_tuple(self, index: tuple) -> NDArray[Any]:
        """Handle tuple indexing with variable subsetting."""
        # Replace variable axis (position 1) with slice(None), fetch all
        # variables, then subset.
        index = list(index)
        if len(index) > 1:
            previous = index[1]
            index[1] = slice(None)
        else:
            previous = slice(None)

        result = self.dataset[tuple(index)]

        # Apply variable selection on axis 1
        # After dataset[tuple(index)], the variable axis may or may not
        # still be at position 1 depending on whether index[0] was scalar.
        if isinstance(index[0], (int,)):
            # scalar date → result is (variables, ensembles, steps, cells)
            result = result[self.indices]
        else:
            # slice/array date → result is (dates, variables, ensembles, steps, cells)
            result = result[:, self.indices]

        # Re-apply the original variable sub-index if it wasn't slice(None)
        if previous != slice(None) and not (isinstance(previous, slice) and previous == slice(None)):
            if isinstance(index[0], (int,)):
                result = result[previous]
            else:
                result = result[:, previous]

        return result
