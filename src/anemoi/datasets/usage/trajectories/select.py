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

from anemoi.utils.dates import frequency_to_string
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

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        """Override to avoid calling self.frequency (trajectory datasets have two frequencies)."""
        action = self.__class__.__name__.lower()
        step_freq = self.step_frequency
        return dict(
            action=action,
            variables=self.variables,
            shape=self.shape,
            base_frequency=frequency_to_string(self.base_frequency),
            step_frequency=frequency_to_string(step_freq) if step_freq is not None else None,
            start_date=str(self.start_date),
            end_date=str(self.end_date),
            base_start_date=str(self.base_start_date),
            base_end_date=str(self.base_end_date),
            forward=self.forward.metadata_specific(),
            **self.forwards_subclass_metadata_specific(),
            **kwargs,
        )

    def dataset_metadata(self) -> dict[str, Any]:
        """Override to avoid calling self.frequency (trajectory datasets have two frequencies)."""
        return dict(
            specific=self.metadata_specific(),
            base_frequency=self.base_frequency,
            step_frequency=self.step_frequency,
            variables=self.variables,
            variables_metadata=self.variables_metadata,
            shape=self.shape,
            dtype=str(self.dtype),
            start_date=str(self.start_date),
            end_date=str(self.end_date),
            base_start_date=str(self.base_start_date),
            base_end_date=str(self.base_end_date),
            name=self.name,
        )
