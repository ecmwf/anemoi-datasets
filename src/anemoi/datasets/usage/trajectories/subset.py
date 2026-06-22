# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Step-axis subsetting wrappers for trajectories datasets."""

import datetime
import logging
from collections.abc import Sequence
from functools import cached_property
from typing import Any
from typing import Union

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.dataset import Shape
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.debug import Source
from anemoi.datasets.usage.debug import debug_indexing
from anemoi.datasets.usage.forwards import Forwards
from anemoi.datasets.usage.gridded.indexing import apply_index_to_slices_changes
from anemoi.datasets.usage.gridded.indexing import expand_list_indexing
from anemoi.datasets.usage.gridded.indexing import index_to_slices
from anemoi.datasets.usage.gridded.indexing import make_slice_or_index_from_list_or_tuple
from anemoi.datasets.usage.gridded.indexing import update_tuple
from anemoi.datasets.usage.trajectories.metadata import trajectory_metadata

LOG = logging.getLogger(__name__)


class StepSubset(Forwards):
    """A view of a trajectories dataset restricted to a subset of steps.

    Slices the step axis (position ``-2``, just before cells) of the
    underlying 5-D ``(dates, variables, ensembles, steps, cells)`` array.
    The date axis (axis 0) is unchanged, so ``__getitem__(n)`` returns an
    array of shape ``(variables, ensembles, len_steps, cells)`` for a scalar
    date index.

    Parameters
    ----------
    dataset : Dataset
        The underlying trajectories dataset.
    step_indices : list of int
        Indices into the original step axis to retain.
    """

    def __init__(self, dataset: Dataset, step_indices: list[int]) -> None:
        super().__init__(dataset)
        self.step_indices = list(step_indices)

    def mutate(self) -> Dataset:
        return self

    @property
    def shape(self) -> Shape:
        """Return the 5-D shape with the step axis narrowed."""
        s = self.forward.shape
        return s[:-2] + (len(self.step_indices), s[-1])

    @property
    def steps(self) -> NDArray[np.timedelta64]:
        """Return the subset of step values."""
        return self.forward.steps[self.step_indices]

    @property
    def step_start(self) -> datetime.timedelta:
        """Return the first step in the subset."""
        return self.steps[0].astype("timedelta64[s]").astype(datetime.timedelta)

    @property
    def step_end(self) -> datetime.timedelta:
        """Return the last step in the subset."""
        return self.steps[-1].astype("timedelta64[s]").astype(datetime.timedelta)

    @property
    def step_frequency(self) -> datetime.timedelta | None:
        """Return the step interval, or None if steps are not uniformly spaced."""
        steps = self.steps
        if len(steps) < 2:
            return None
        diffs = np.diff(steps)
        if np.all(diffs == diffs[0]):
            return diffs[0].astype("timedelta64[s]").astype(datetime.timedelta)
        return None

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Return data for the given date index, sliced to the selected steps."""
        # Step axis is at position -2 (just before cells).
        # Scalar n: forward returns (vars, ensembles, steps, cells) — 4-D.
        # Slice/array n: forward returns (dates, vars, ensembles, steps, cells) — 5-D.
        # ``[..., step_indices, :]`` targets the step axis in both cases.
        #
        # Tuple indices cannot be passed through to the underlying dataset:
        # any element beyond axis 0 could consume the step axis before the
        # step selection is applied, or index the un-subsetted step axis
        # directly.  Select along axis 0 first (which applies the step
        # subsetting), then apply the remaining indices with numpy semantics.
        if isinstance(n, tuple):
            result = self[n[0]]
            rest = n[1:]
            if not rest:
                return result
            if isinstance(n[0], (int, np.integer)):
                return result[rest]
            return result[(slice(None),) + rest]
        return self.forward[n][..., self.step_indices, :]

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
        """Delegate to the underlying dataset, which resolves a default delta
        from the step frequency for trajectories.
        """
        return self.forward.statistics_tendencies(delta)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return {"step_indices": self.step_indices}

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        return super().metadata_specific(**trajectory_metadata(self), **kwargs)

    def dataset_metadata(self) -> dict[str, Any]:
        md = super().dataset_metadata()
        md.update(trajectory_metadata(self))
        return md

    def tree(self) -> Node:
        return Node(self, [self.forward.tree()])


class SingleStepView(Forwards):
    """A gridded-compatible view of a trajectories dataset for a single step.

    Selects one step from the step axis (position ``-2``) and drops it,
    producing an array of shape ``(dates, variables, ensembles, cells)`` —
    identical to what a gridded dataset returns.  ``dates`` and ``frequency``
    are defined (valid times ``base_dates + step`` and ``base_frequency``)
    so that gridded wrappers needing a single time axis can be stacked on
    top of this view.

    Parameters
    ----------
    dataset : Dataset
        The underlying trajectories dataset.
    step_index : int
        Index into the step axis to select.
    """

    def __init__(self, dataset: Dataset, step_index: int) -> None:
        super().__init__(dataset)
        self.step_index = step_index

    def mutate(self) -> Dataset:
        return self

    @property
    def shape(self) -> Shape:
        """Return the 4-D shape ``(dates, variables, ensembles, cells)``."""
        s = self.forward.shape
        return s[:-2] + (s[-1],)

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Return the valid times of the view: ``base_dates + step``."""
        return self.forward.base_dates + self.forward.steps[self.step_index]

    @property
    def frequency(self) -> datetime.timedelta:
        """Return the interval between consecutive valid times (the base-date frequency)."""
        return self.forward.base_frequency

    @property
    def start_date(self) -> np.datetime64:
        """Return the first valid time of the view."""
        return self.dates[0]

    @property
    def end_date(self) -> np.datetime64:
        """Return the last valid time of the view."""
        return self.dates[-1]

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Return data for the given date index with the step axis removed."""
        # Scalar n: (vars, ensembles, steps, cells) → (vars, ensembles, cells).
        # Slice/array n: (dates, vars, ensembles, steps, cells) → (dates, vars, ensembles, cells).
        #
        # Tuple indices cannot be passed through: the underlying array is
        # 5-D while this view is 4-D, so any element beyond axis 0 would be
        # applied to the wrong axis.  Select along axis 0 first (which drops
        # the step axis), then apply the remaining indices.
        if isinstance(n, tuple):
            result = self[n[0]]
            rest = n[1:]
            if not rest:
                return result
            if isinstance(n[0], (int, np.integer)):
                return result[rest]
            return result[(slice(None),) + rest]
        return self.forward[n][..., self.step_index, :]

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
        """Delegate to the underlying dataset, which resolves a default delta
        from the step frequency for trajectories.
        """
        return self.forward.statistics_tendencies(delta)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return {"step_index": self.step_index}

    def tree(self) -> Node:
        return Node(self, [self.forward.tree()])


class Subset(Forwards):
    """Select a subset of base dates from a trajectories dataset.

    Mirrors the gridded ``Subset`` but operates on ``base_dates`` instead
    of ``dates``, and provides trajectory-compatible metadata (envelope
    ``start_date``/``end_date``, ``base_frequency``, etc.).

    Parameters
    ----------
    dataset : Dataset
        The underlying trajectories dataset.
    indices : Sequence of int
        Indices into the base-date axis to retain.
    reason : dict
        Provenance metadata for the subsetting operation.
    """

    def __init__(self, dataset: Union[Dataset, "Subset"], indices: Sequence[int], reason: dict[str, Any]) -> None:
        while isinstance(dataset, Subset):
            indices = [dataset.indices[i] for i in indices]
            dataset = dataset.dataset

        self.dataset: Dataset = dataset
        self.indices: list[int] = list(indices)
        self.reason: dict[str, Any] = {k: v for k, v in reason.items() if v is not None}

        super().__init__(dataset)

    def clone(self, dataset: Dataset) -> Dataset:
        return self.__class__(dataset, self.indices, self.reason).mutate()

    def mutate(self) -> Dataset:
        return self.forward.swap_with_parent(parent=self)

    @debug_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        if isinstance(n, tuple):
            return self._get_tuple(n)
        if isinstance(n, slice):
            return self._get_slice(n)
        assert n >= 0, n
        return self.dataset[self.indices[n]]

    @debug_indexing
    def _get_slice(self, s: slice) -> NDArray[Any]:
        indices = [self.indices[i] for i in range(*s.indices(len(self)))]
        indices = make_slice_or_index_from_list_or_tuple(indices)
        if isinstance(indices, slice):
            return self.dataset[indices]
        return np.stack([self.dataset[i] for i in indices])

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, n) -> NDArray[Any]:
        index, changes = index_to_slices(n, self.shape)
        indices = [self.indices[i] for i in range(*index[0].indices(len(self)))]
        indices = make_slice_or_index_from_list_or_tuple(indices)
        index, _ = update_tuple(index, 0, indices)
        result = self.dataset[index]
        result = apply_index_to_slices_changes(result, changes)
        return result

    def __len__(self) -> int:
        return len(self.indices)

    @cached_property
    def shape(self) -> Shape:
        return (len(self),) + self.dataset.shape[1:]

    # -- Base-date axis (subsetted) --

    @cached_property
    def base_dates(self) -> NDArray[np.datetime64]:
        return self.dataset.base_dates[self.indices]

    @property
    def base_start_date(self) -> np.datetime64:
        return self.base_dates[0]

    @property
    def base_end_date(self) -> np.datetime64:
        return self.base_dates[-1]

    @cached_property
    def base_frequency(self) -> datetime.timedelta:
        dates = self.base_dates
        if len(dates) < 2:
            raise ValueError(f"Cannot determine base_frequency with fewer than two base dates ({dates}).")
        return frequency_to_timedelta(dates[1].astype(object) - dates[0].astype(object))

    # -- Envelope --

    @property
    def start_date(self) -> np.datetime64:
        return self.base_dates[0] + self.dataset.steps[0]

    @property
    def end_date(self) -> np.datetime64:
        return self.base_dates[-1] + self.dataset.steps[-1]

    # -- dates/frequency fail (forwarded from underlying trajectories dataset) --

    @cached_property
    def missing(self) -> set[int]:
        missing = self.dataset.missing
        result: set[int] = set()
        for j, i in enumerate(self.indices):
            if i in missing:
                result.add(j)
        return result

    def source(self, index: int) -> Source:
        return Source(self, index, self.forward.source(index))

    def __repr__(self) -> str:
        return f"Subset({self.dataset}, {self.base_dates[0]}...{self.base_dates[-1]})"

    def tree(self) -> Node:
        return Node(self, [self.dataset.tree()], **self.reason)

    def _dates_to_indices(self, start, end) -> list[int]:
        """Strict envelope filtering on the subsetted base dates."""
        from anemoi.datasets.usage.misc import as_first_date
        from anemoi.datasets.usage.misc import as_last_date

        base_dates = self.base_dates
        step_start = self.dataset.steps[0]
        step_end = self.dataset.steps[-1]

        valid_starts = base_dates + step_start
        valid_ends = base_dates + step_end

        if start is not None:
            start = as_first_date(start, valid_starts)
        else:
            start = valid_starts[0]

        if end is not None:
            end = as_last_date(end, valid_ends)
        else:
            end = valid_ends[-1]

        base_min = start - step_start
        base_max = end - step_end

        return [i for i, d in enumerate(base_dates) if base_min <= d <= base_max]

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
        """Delegate to the underlying dataset, which resolves a default delta
        from the step frequency for trajectories.
        """
        return self.forward.statistics_tendencies(delta)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return {"indices": self.indices, "reason": self.reason}

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        return super().metadata_specific(**trajectory_metadata(self), **kwargs)

    def dataset_metadata(self) -> dict[str, Any]:
        md = super().dataset_metadata()
        md.update(trajectory_metadata(self))
        return md
