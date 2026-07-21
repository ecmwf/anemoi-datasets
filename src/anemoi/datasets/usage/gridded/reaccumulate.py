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

from anemoi.datasets import MissingDateError

from ..dataset import Dataset
from ..dataset import FullIndex
from ..debug import Node
from ..debug import debug_indexing
from ..forwards import Forwards

LOG = logging.getLogger(__name__)


class Reaccumulate(Forwards):
    """Resample selected variables to a windowed accumulation.

    Each output row ``n`` is anchored at raw index ``m = n * stride``. The
    row's window spans the ``window_length`` raw timesteps ``[m, m +
    window_length)``, and its own labelled date/raw-timestep sits at offset
    ``-window[0]`` into that window. Variables listed in
    ``variables`` are *summed* over the window; every other variable is taken
    unchanged from the row's own labelled raw timestep.

    ``stride`` controls whether this collapses the time axis or preserves it:

    - ``stride == window_length`` -- windows are non-overlapping blocks, one
      per output row, and the time axis collapses to a coarser frequency
      (``stride * native frequency``). Use :meth:`from_step` for this shape,
      e.g. re-accumulating hourly ``tp`` into a 6-hourly total.
    - ``stride == 1`` -- windows slide by one raw timestep between rows, and
      every native timestep is kept in the output. Use :meth:`from_window`
      for this shape, e.g. a sliding 6-hourly accumulation of ``tp``
      evaluated at every 3-hourly step for an input/target training scheme.

    Other strides are accepted but not exercised by either convenience
    constructor above.
    """

    def __init__(self, dataset: Dataset, window: tuple[int, int, str], stride: int, variables: list[str]) -> None:
        """Initialize the Reaccumulate class.

        Parameters
        ----------
        dataset : Dataset
            The dataset to resample.
        window : (int, int, str)
            The accumulation window (start, end, 'freq'): both bounds
            inclusive, in raw timesteps, relative to each row's own labelled
            date, and 'freq' is currently the only supported unit.
        stride : int
            Number of raw timesteps between the anchors of consecutive
            output rows. ``stride == window_length`` gives non-overlapping
            blocks (frequency-collapsing); ``stride == 1`` gives a sliding
            window (frequency-preserving).
        variables : list of str
            Names of the variables to sum over each window. Any other
            variable is taken from its own raw timestep, unchanged.
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
        if self.i_start < 0:
            raise ValueError(f"Window start must be negative, got {window}")
        if self.i_end <= 0:
            raise ValueError(f"Window end must be positive, got {window}")

        self.window_str = f"-{self.i_start}-to-{self.i_end}"
        self.window_length = self.i_start + self.i_end

        if not isinstance(stride, int) or stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride}")
        self.stride = stride

        name_to_index = dataset.name_to_index
        unknown = [v for v in variables if v not in name_to_index]
        if unknown:
            raise ValueError(f"reaccumulate: unknown variable(s) {unknown}, available: {list(name_to_index)}")

        self._check_accumulation_periods(dataset, variables, self.window_length * dataset.frequency)

        self.accumulated_variables: list[str] = list(variables)
        self.accumulated_indices: list[int] = [name_to_index[v] for v in variables]

    @classmethod
    def from_step(cls, dataset: Dataset, step: int, variables: list[str]) -> "Reaccumulate":
        """Non-overlapping block accumulation: collapse the time axis to a
        coarser frequency (``step`` raw timesteps per output row), summing
        ``variables`` over each block.

        Parameters
        ----------
        dataset : Dataset
            The dataset to resample.
        step : int
            Number of raw timesteps per output timestep (i.e. the ratio
            between the requested frequency and the dataset's native
            frequency).
        variables : list of str
            Names of the variables to sum over each block.
        """
        if not isinstance(step, int) or step < 1:
            raise ValueError(f"step must be a positive integer, got {step}")
        return cls(dataset, (-(step - 1), 0, "freq"), step, variables)

    @classmethod
    def from_window(cls, dataset: Dataset, window: tuple[int, int, str], variables: list[str]) -> "Reaccumulate":
        """Sliding-window accumulation: keep the dataset at its native
        frequency, summing ``variables`` over a window of raw timesteps
        around every row.

        Parameters
        ----------
        dataset : Dataset
            The dataset to resample.
        window : (int, int, str)
            The rolling window (start, end, 'freq'), see the class
            docstring. For a trailing accumulation ending at each output
            row's own labelled date (e.g. combining two 3h increments into a
            6h total), use ``(-1, 0, "freq")``.
        variables : list of str
            Names of the variables to sum over the window.
        """
        return cls(dataset, window, 1, variables)

    @staticmethod
    def _check_accumulation_periods(dataset: Dataset, variables: list[str], target_period: datetime.timedelta) -> None:
        """Fail if a variable's own accumulation period (from the dataset's
        metadata) is not a proper factor of ``target_period`` (the period
        being summed over, i.e. ``window_length * native frequency``), since
        summing raw timesteps whose accumulation window doesn't evenly tile
        that period would silently produce an incorrect total.

        Variables with no recorded accumulation period (missing/empty
        metadata) are skipped.
        """
        from anemoi.transform.variables import Variable

        metadata = dataset.variables_metadata
        for var in variables:
            meta = metadata.get(var)
            if not meta:
                continue

            period = Variable.from_dict(var, meta).period
            if not period:
                continue

            if target_period % period != datetime.timedelta(0):
                raise ValueError(
                    f"variable {var!r} has an accumulation period of {period}, which is not a proper "
                    f"factor of the period being summed over ({target_period})."
                )

    def __len__(self) -> int:
        return (len(self.forward) - self.window_length) // self.stride + 1

    @property
    def shape(self):
        shape = list(self.forward.shape)
        shape[0] = len(self)
        return tuple(shape)

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Dates of each row, taken as the labelled raw date within its window."""
        indices = self.i_start + self.stride * np.arange(len(self))
        return self.forward.dates[indices]

    @cached_property
    def frequency(self) -> datetime.timedelta:
        """Effective frequency of the resampled dataset: the raw-timestep
        spacing between consecutive output rows' labelled dates
        (``stride * native frequency``, regardless of ``window_length``).
        """
        return self.stride * self.forward.frequency

    def _raise_if_missing_in_window(self, n: int, forward_missing: set[int]) -> None:
        m = n * self.stride
        for j in range(m, m + self.window_length):
            if j in forward_missing:
                raise MissingDateError(
                    f"Reaccumulate window for date {self.forward.dates[m + self.i_start]} "
                    f"overlaps with missing forward index {j} (date {self.forward.dates[j]})"
                )

    def _get_one(self, n: int, rest: tuple) -> NDArray[Any]:
        if n < 0:
            n = len(self) + n
        if n < 0 or n >= len(self):
            raise IndexError(f"Index out of range: {n}")

        forward_missing = self.forward.missing
        if forward_missing:
            self._raise_if_missing_in_window(n, forward_missing)

        m = n * self.stride
        label_index = m + self.i_start
        var_index = rest[0] if rest else slice(None)
        other_rest = rest[1:] if rest else ()

        # Non-accumulated variables only need the raw timestep matching this
        # row's labelled date, and any variable/grid subsetting is pushed
        # down into the forward fetch. 
        result = self.forward[(slice(label_index, label_index + 1), var_index) + other_rest][0]

        if not self.accumulated_indices:
            return result

        n_vars = self.forward.shape[1]
        accumulated_set = set(self.accumulated_indices)

        if isinstance(var_index, int):
            var = var_index if var_index >= 0 else var_index + n_vars
            if var not in accumulated_set:
                return result
            window = self.forward[(slice(m, m + self.window_length), var) + other_rest]
            return np.sum(window, axis=0)

        if isinstance(var_index, slice):
            requested = list(range(*var_index.indices(n_vars)))
        else:
            values = var_index.tolist() if hasattr(var_index, "tolist") else var_index
            requested = [v if v >= 0 else v + n_vars for v in values]

        acc_positions = [p for p, v in enumerate(requested) if v in accumulated_set]
        if not acc_positions:
            return result

        acc_vars = [requested[p] for p in acc_positions]
        window = self.forward[(slice(m, m + self.window_length), acc_vars) + other_rest]
        result = np.array(result, copy=True)
        result[acc_positions] = np.sum(window, axis=0)
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
        """A row is missing if any raw timestep its window depends on is missing."""
        forward_missing = self.forward.missing
        if not forward_missing:
            return set()

        result = set()
        for n in range(len(self)):
            m = n * self.stride
            if any(i in forward_missing for i in range(m, m + self.window_length)):
                result.add(n)
        return result

    def tree(self) -> Node:
        return Node(
            self,
            [self.forward.tree()],
            window=self.window_str,
            stride=self.stride,
            reaccumulate=self.accumulated_variables,
        )

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return dict(window=self.window_str, stride=self.stride, reaccumulate=self.accumulated_variables)
