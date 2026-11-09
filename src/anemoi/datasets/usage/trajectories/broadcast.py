# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Broadcast a gridded dataset onto a trajectories layout.

A gridded dataset has the 4-D layout ``(dates, variables, ensembles, cells)``
and no step axis.  :class:`GriddedAsTrajectory` presents it as a 5-D
trajectories dataset ``(base_dates, variables, ensembles, steps, cells)`` by
looking the gridded fields up on **valid time**: the value placed at trajectory
position ``(base date b, step s)`` is the gridded field at valid time
``b + s``.  The same gridded field is therefore duplicated wherever two
``(base, step)`` pairs share a valid time, and every required valid time must be
present in the gridded dataset (otherwise construction fails).

This wrapper is an implementation detail of :func:`trajectory_join`; it is not
meant to be opened directly.
"""

import datetime
import logging
from functools import cached_property
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.dataset import Shape
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.forwards import Forwards
from anemoi.datasets.usage.trajectories.metadata import trajectory_metadata

LOG = logging.getLogger(__name__)


class GriddedAsTrajectory(Forwards):
    """View a gridded dataset as a trajectories dataset by valid-time lookup.

    Parameters
    ----------
    dataset : Dataset
        The underlying gridded dataset (4-D layout).
    base_dates : NDArray[np.datetime64]
        The target base dates, taken from the trajectory template.
    steps : NDArray[np.timedelta64]
        The target forecast steps, taken from the trajectory template.
    """

    def __init__(
        self,
        dataset: Dataset,
        base_dates: NDArray[np.datetime64],
        steps: NDArray[np.timedelta64],
    ) -> None:
        super().__init__(dataset)
        self._base_dates = base_dates.astype("datetime64[s]")
        self._steps = steps.astype("timedelta64[s]")
        self._index_map = self._build_index_map()

    def _build_index_map(self) -> NDArray[np.int64]:
        """Map each ``(base date, step)`` pair to a row in the gridded dataset.

        Returns
        -------
        NDArray[np.int64]
            An ``(n_base_dates, n_steps)`` array of indices into the gridded
            dataset's date axis.

        Raises
        ------
        ValueError
            If any required valid time is not present in the gridded dataset.
        """
        gridded_dates = self.forward.dates.astype("datetime64[s]")
        lookup = {d: i for i, d in enumerate(gridded_dates.tolist())}

        # valid[i, j] = base_dates[i] + steps[j]
        valid = self._base_dates[:, None] + self._steps[None, :]

        index_map = np.empty(valid.shape, dtype=np.int64)
        missing: set[np.datetime64] = set()
        for i in range(valid.shape[0]):
            for j in range(valid.shape[1]):
                t = valid[i, j].tolist()
                k = lookup.get(t)
                if k is None:
                    missing.add(valid[i, j])
                else:
                    index_map[i, j] = k

        if missing:
            sample = sorted(str(np.datetime64(m, "s")) for m in missing)[:10]
            raise ValueError(
                f"{self.forward} does not cover {len(missing)} valid time(s) required to broadcast "
                f"it onto the trajectory (base_date + step). First missing: {sample}"
            )

        return index_map

    def mutate(self) -> Dataset:
        return self

    # ------------------------------------------------------------------
    # Shape and indexing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of base dates."""
        return len(self._base_dates)

    @cached_property
    def shape(self) -> Shape:
        """Return the 5-D shape ``(base_dates, variables, ensembles, steps, cells)``."""
        variables, ensembles, cells = self.forward.shape[1], self.forward.shape[2], self.forward.shape[3]
        return (len(self), variables, ensembles, len(self._steps), cells)

    def _single(self, i: int) -> NDArray[Any]:
        """Return the broadcast field for a single base date.

        Parameters
        ----------
        i : int
            Index along the base-date axis.

        Returns
        -------
        NDArray[Any]
            An array of shape ``(variables, ensembles, steps, cells)``.
        """
        # Look up the gridded field (variables, ensembles, cells) for the valid
        # time of every step, and stack them along the step axis (position -2).
        fields = [self.forward[int(k)] for k in self._index_map[i]]
        return np.stack(fields, axis=-2)

    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Return data for the given base-date index.

        Parameters
        ----------
        n : int, slice, list, tuple or ndarray
            Index into the base-date axis (and, for tuples, the trailing axes).

        Returns
        -------
        NDArray[Any]
            The requested slice of the 5-D array.
        """
        if isinstance(n, tuple):
            # Resolve the base-date axis first (which inserts the step axis),
            # then apply the remaining indices with plain numpy semantics — the
            # underlying gridded array is 4-D, so a tuple cannot be forwarded.
            result = self[n[0]]
            rest = n[1:]
            if not rest:
                return result
            if isinstance(n[0], (int, np.integer)):
                return result[rest]
            return result[(slice(None),) + rest]

        if isinstance(n, slice):
            return np.stack([self._single(i) for i in range(*n.indices(len(self)))])

        if isinstance(n, (list, np.ndarray)):
            return np.stack([self._single(int(i)) for i in n])

        return self._single(int(n))

    # ------------------------------------------------------------------
    # Base-date axis
    # ------------------------------------------------------------------

    @property
    def base_dates(self) -> NDArray[np.datetime64]:
        """Return the base dates."""
        return self._base_dates

    @property
    def base_start_date(self) -> np.datetime64:
        """Return the first base date."""
        return self._base_dates[0]

    @property
    def base_end_date(self) -> np.datetime64:
        """Return the last base date."""
        return self._base_dates[-1]

    @property
    def base_frequency(self) -> datetime.timedelta:
        """Return the interval between consecutive base dates."""
        dates = self._base_dates
        if len(dates) < 2:
            raise ValueError(f"Cannot determine base_frequency with fewer than two base dates ({dates}).")
        return frequency_to_timedelta(dates[1].astype(object) - dates[0].astype(object))

    # ------------------------------------------------------------------
    # Step axis
    # ------------------------------------------------------------------

    @property
    def steps(self) -> NDArray[np.timedelta64]:
        """Return the forecast steps."""
        return self._steps

    @property
    def step_start(self) -> datetime.timedelta:
        """Return the first forecast step."""
        return self._steps[0].astype("timedelta64[s]").astype(datetime.timedelta)

    @property
    def step_end(self) -> datetime.timedelta:
        """Return the last forecast step."""
        return self._steps[-1].astype("timedelta64[s]").astype(datetime.timedelta)

    @property
    def step_frequency(self) -> datetime.timedelta | None:
        """Return the step interval, or None if steps are not uniformly spaced."""
        if len(self._steps) < 2:
            return None
        diffs = np.diff(self._steps)
        if np.all(diffs == diffs[0]):
            return diffs[0].astype("timedelta64[s]").astype(datetime.timedelta)
        return None

    # ------------------------------------------------------------------
    # Envelope (valid-time range) and disabled single time axis
    # ------------------------------------------------------------------

    @property
    def start_date(self) -> np.datetime64:
        """Return the earliest valid time: first base date + first step."""
        return self._base_dates[0] + self._steps[0]

    @property
    def end_date(self) -> np.datetime64:
        """Return the latest valid time: last base date + last step."""
        return self._base_dates[-1] + self._steps[-1]

    @property
    def dates(self) -> NDArray[np.datetime64]:
        """Trajectories datasets do not have a single ``dates`` array."""
        raise AttributeError(
            "Trajectories datasets have two time axes. "
            "Use 'base_dates' for analysis times and 'steps' for forecast lead times."
        )

    @property
    def frequency(self) -> datetime.timedelta | None:
        """Trajectories datasets have two frequencies, so ``frequency`` is None."""
        return None

    # ------------------------------------------------------------------
    # Missing base dates
    # ------------------------------------------------------------------

    @cached_property
    def missing(self) -> set[int]:
        """Return base-date indices whose valid-time lookups hit a missing gridded row."""
        gridded_missing = self.forward.missing
        if not gridded_missing:
            return set()
        return {i for i in range(len(self)) if set(self._index_map[i].tolist()) & gridded_missing}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
        """Return the underlying gridded dataset's tendencies.

        The broadcast gridded variables carry the gridded dataset's own
        tendencies (computed along the gridded time axis at its own frequency).
        The trajectory step-frequency ``delta`` propagated by the join does not
        apply to them, so it is ignored and the gridded dataset's default is
        used — this never raises for a lack of a step-frequency tendencies key.
        """
        return self.forward.statistics_tendencies()

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return {"broadcast": "gridded_as_trajectory"}

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        return super().metadata_specific(**trajectory_metadata(self), **kwargs)

    def dataset_metadata(self) -> dict[str, Any]:
        md = super().dataset_metadata()
        md.update(trajectory_metadata(self))
        return md

    def tree(self) -> Node:
        return Node(self, [self.forward.tree()])

    def __repr__(self) -> str:
        return f"GriddedAsTrajectory({self.forward})"
