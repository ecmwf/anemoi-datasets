# (C) Copyright 2026 Anemoi contributors.
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

import numpy as np
from numpy.typing import NDArray

from ..dataset import Dataset
from ..dataset import FullIndex
from ..debug import Node
from ..debug import debug_indexing
from ..forwards import Forwards

LOG = logging.getLogger(__name__)


class Reaccumulate(Forwards):
    """Resample a dataset to a coarser, but still exact, multiple of its
    native frequency, re-accumulating selected variables instead of just
    picking a single raw timestep.

    The dataset is split into non-overlapping blocks of ``step`` raw
    timesteps. For each block, variables listed in ``variables`` (e.g.
    ``tp``) are summed over the block, giving the accumulation over
    the coarser period. All other variables are taken from the last raw
    timestep of the block. Output dates are therefore the *end* of each
    block, matching the usual convention that an accumulated field is
    labelled with the end of its accumulation period.
    """

    def __init__(self, dataset: Dataset, step: int, variables: list[str]) -> None:
        """Initialize the Reaccumulate class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to resample.
        step : int
            Number of raw timesteps per output timestep (i.e. the ratio
            between the requested frequency and the dataset's native
            frequency).
        variables : list of str
            Names of the variables to sum over each block. Any other
            variable is taken from the last raw timestep of the block.
        """
        super().__init__(dataset)

        if not isinstance(step, int) or step < 1:
            raise ValueError(f"step must be a positive integer, got {step}")
        self.step = step

        name_to_index = dataset.name_to_index
        unknown = [v for v in variables if v not in name_to_index]
        if unknown:
            raise ValueError(f"reaccumulate: unknown variable(s) {unknown}, available: {list(name_to_index)}")

        self.accumulated_variables: list[str] = list(variables)
        self.accumulated_indices: list[int] = [name_to_index[v] for v in variables]

    def __len__(self) -> int:
        return len(self.forward) // self.step

    @property
    def shape(self):
        shape = list(self.forward.shape)
        shape[0] = len(self)
        return tuple(shape)

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Dates of each block, taken as the last raw date in the block."""
        dates = self.forward.dates
        return dates[self.step - 1 :: self.step][: len(self)]

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Effective frequency of the resampled dataset (step * native frequency)."""
        return self.step * self.forward.frequency

    def _get_one(self, n: int, rest: tuple) -> NDArray[Any]:
        if n < 0:
            n = len(self) + n
        if n < 0 or n >= len(self):
            raise IndexError(f"Index out of range: {n}")

        start = n * self.step
        var_index = rest[0] if rest else slice(None)
        other_rest = rest[1:] if rest else ()

        # Non-accumulated variables only need the last raw timestep of the
        # block, and any variable/grid subsetting is pushed down into the
        # forward fetch.
        result = self.forward[(start + self.step - 1, var_index) + other_rest]

        if not self.accumulated_indices:
            return result

        n_vars = self.forward.shape[1]
        accumulated_set = set(self.accumulated_indices)

        if isinstance(var_index, int):
            var = var_index if var_index >= 0 else var_index + n_vars
            if var not in accumulated_set:
                return result
            block = self.forward[(slice(start, start + self.step), var) + other_rest]
            return np.sum(block, axis=0)

        if isinstance(var_index, slice):
            requested = list(range(*var_index.indices(n_vars)))
        else:
            values = var_index.tolist() if hasattr(var_index, "tolist") else var_index
            requested = [v if v >= 0 else v + n_vars for v in values]

        acc_positions = [p for p, v in enumerate(requested) if v in accumulated_set]
        if not acc_positions:
            return result

        acc_vars = [requested[p] for p in acc_positions]
        block = self.forward[(slice(start, start + self.step), acc_vars) + other_rest]
        result = np.array(result, copy=True)
        result[acc_positions] = np.sum(block, axis=0)
        return result

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        if isinstance(n, tuple):
            first, rest = n[0], n[1:]
        else:
            first, rest = n, ()

        if isinstance(first, slice):
            first = list(range(*first.indices(len(self))))

        if isinstance(first, (list, tuple)):
            return np.stack([self._get_one(i, rest) for i in first])

        return self._get_one(first, rest)

    @cached_property
    def missing(self) -> set[int]:
        """A block is missing if any raw timestep it is built from is missing."""
        forward_missing = self.forward.missing
        if not forward_missing:
            return set()

        result = set()
        for n in range(len(self)):
            start = n * self.step
            if any(i in forward_missing for i in range(start, start + self.step)):
                result.add(n)
        return result

    def tree(self) -> Node:
        return Node(self, [self.forward.tree()], step=self.step, reaccumulate=self.accumulated_variables)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return dict(step=self.step, reaccumulate=self.accumulated_variables)
