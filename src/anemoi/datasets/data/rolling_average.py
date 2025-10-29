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

from anemoi.datasets.data.indexing import expand_list_indexing

from .dataset import Dataset
from .dataset import FullIndex
from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards

LOG = logging.getLogger(__name__)


class RollingAverage(Forwards):
    """A class to represent a dataset with interpolated frequency."""

    def __init__(self, dataset: Dataset, window: str | tuple[int, int, str]) -> None:
        """Initialize the RollingAverage class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be averaged with a rolling window.
        window : (int, int, str)
            The rolling average window (start, end, 'freq').
            'freq' means the window is in number of time steps in the dataset.
            Both start and end are inclusive, i.e. window = (-2, 2, 'freq') means a window of 5 time steps.
            For now, only 'freq' is supported, in the future other units may be supported.
            Windows such as "[-2h, +2h]" are not supported yet.
        """
        super().__init__(dataset)
        if not (isinstance(window, (list, tuple)) and len(window) == 3):
            raise ValueError(f"Window must be (int, int, str), got {window}")
        if not isinstance(window[0], int) or not isinstance(window[1], int) or not isinstance(window[2], str):
            raise ValueError(f"Window must be (int, int, str), got {window}")
        if window[2] not in ["freq", "frequency"]:
            raise NotImplementedError(f"Window must be (int, int, 'freq'), got {window}")

        # window = (0, 0, 'freq') means no change
        self.i_start = -window[0]
        self.i_end = window[1] + 1
        if self.i_start <= 0:
            raise ValueError(f"Window start must be negative, got {window}")
        if self.i_end <= 0:
            raise ValueError(f"Window end must be positive, got {window}")

        self.window_str = f"-{self.i_start}-to-{self.i_end}"

    @property
    def shape(self):
        shape = list(self.forward.shape)
        shape[0] = len(self)
        return tuple(shape)

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        def f(array):
            return np.nanmean(array, axis=0)

        if isinstance(n, slice):
            n = (n,)

        if isinstance(n, tuple):
            first = n[0]
            if len(n) > 1:
                rest = n[1:]
            else:
                rest = ()

            if isinstance(first, int):
                slice_ = slice(first, first + self.i_start + self.i_end)
                data = self.forward[(slice_,) + rest]
                return f(data)

            if isinstance(first, slice):
                first = list(range(first.start or 0, first.stop or len(self), first.step or 1))

            if isinstance(first, (list, tuple)):
                first = [i if i >= 0 else len(self) + i for i in first]
                if any(i >= len(self) for i in first):
                    raise IndexError(f"Index out of range: {first}")
                slices = [slice(i, i + self.i_start + self.i_end) for i in first]
                data = [self.forward[(slice_,) + rest] for slice_ in slices]
                res = [f(d) for d in data]
                return np.array(res)

            assert False, f"Expected int, slice, list or tuple as first element of tuple, got {type(first)}"

        assert isinstance(n, int), f"Expected int, slice, tuple, got {type(n)}"

        if n < 0:
            n = len(self) + n
        if n >= len(self):
            raise IndexError(f"Index out of range: {n}")

        slice_ = slice(n, n + self.i_start + self.i_end)
        data = self.forward[slice_]
        return f(data)

    def __len__(self) -> int:
        return len(self.forward) - (self.i_end + self.i_start - 1)

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Get the interpolated dates."""
        dates = self.forward.dates
        return dates[self.i_start : len(dates) - self.i_end + 1]

    def tree(self) -> Node:
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
