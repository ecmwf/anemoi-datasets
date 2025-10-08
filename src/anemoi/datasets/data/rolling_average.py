# (C) Copyright 2025 Anemoi contributors.
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

import numpy as np
from numpy.typing import NDArray

from .dataset import Dataset
from .dataset import FullIndex
from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards

LOG = logging.getLogger(__name__)


class RollingAverage(Forwards):
    """A class to represent a dataset with interpolated frequency."""

    def __init__(self, dataset: Dataset, window) -> None:
        """Initialize the InterpolateFrequency class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be interpolated.
        frequency : str
            The interpolation frequency.
        """
        super().__init__(dataset)
        if not (isinstance(window, (list, tuple)) and len(window) == 3):
            raise ValueError(f"Window must be (int, int, str), got {window}")
        if not isinstance(window[0], int) or not isinstance(window[1], int) or not isinstance(window[2], str):
            raise ValueError(f"Window must be (int, int, str), got {window}")
        if window[2] not in ["freq", "frequency"]:
            raise NotImplementedError(f"Window must be (int, int, 'freq'), got {window}")

        self.i_start = -window[0]
        self.i_end = window[1]
        # i_start, i_end = 0, 1 means no change
        if self.i_start <= 0:
            raise ValueError(f"Window start must be negative, got {window}")
        if self.i_end <= 0:
            raise ValueError(f"Window end must be positive, got {window}")

        self.window_str = f"-{self.i_start}-to-{self.i_end}"

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self._get_slice(n)

        slice_ = slice(n, n + self.i_start + self.i_end)
        data = self.forward[slice_]
        return np.nanmean(data, axis=0)

    def __len__(self) -> int:
        """Get the length of the interpolated dataset.

        Returns
        -------
        int
            The length of the interpolated dataset.
        """
        return self.forward.__len__() - (self.i_end + self.i_start - 1)

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Get the interpolated dates."""
        dates = self.forward.dates
        return dates[self.i_start : len(dates) - self.i_end + 1]

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], window=self.window_str)

    @cached_property
    def missing(self) -> set[int]:
        """Get the missing data indices."""
        result = []

        for i in self.forward.missing:
            for j in range(0, self.i_end + self.i_start):
                result.append(i + j)

        result = {x for x in result if x < self._len}
        return result

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return {}
