# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os
from functools import cached_property
from typing import Any

import numpy as np
import zarr
from anemoi.utils.dates import frequency_to_timedelta
from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.dataset import FullIndex
from anemoi.datasets.usage.dataset import Shape
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.debug import Source
from anemoi.datasets.usage.debug import debug_indexing
from anemoi.datasets.usage.gridded.indexing import expand_list_indexing

from ..store import ZarrStore

LOG = logging.getLogger(__name__)


class TrajectoriesZarr(ZarrStore):
    """A zarr dataset with a 5-D trajectories layout.

    On-disk array dimensions: ``(dates, variables, ensembles, steps, cells)``.
    The step axis is placed at position ``-2`` so that the first three axes
    match the gridded layout (dates, variables, ensembles) and the cell axis
    remains last.

    Parameters
    ----------
    group : zarr.Group
        The opened zarr group.
    path : str, optional
        Human-readable path label for repr and error messages.
    """

    def __init__(self, group: zarr.Group, path: str = None) -> None:
        super().__init__(group, path=path)

    def mutate(self) -> Dataset:
        """Wrap with :class:`TrajectoriesZarrWithMissingDates` if the store
        records any missing base dates; return ``self`` otherwise.
        """
        if len(self.store.attrs.get("missing_dates", [])):
            LOG.warning(f"Dataset {self} has missing base dates")
            return TrajectoriesZarrWithMissingDates(self.store, self.path)
        return self

    def __len__(self) -> int:
        """Return the number of base dates in the dataset."""
        return self.data.shape[0]

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Return data for the given date index.

        Parameters
        ----------
        n : int, slice, or tuple
            Index into the date axis.

        Returns
        -------
        NDArray[Any]
            An array of shape ``(variables, ensembles, steps, cells)`` for a single integer.
        """
        return self.data[n]

    # ------------------------------------------------------------------
    # Coordinate arrays
    # ------------------------------------------------------------------

    @cached_property
    def dates(self) -> NDArray[np.datetime64]:
        """Trajectories datasets do not have a single ``dates`` array.

        Use ``base_dates`` for the analysis times or ``steps`` for the
        forecast lead times.
        """
        raise AttributeError(
            "Trajectories datasets have two time axes. "
            "Use 'base_dates' for analysis times and 'steps' for forecast lead times."
        )

    # ------------------------------------------------------------------
    # Base-date axis
    # ------------------------------------------------------------------

    @cached_property
    def base_dates(self) -> NDArray[np.datetime64]:
        """Return the base dates (forecast initialisation times) of the dataset."""
        return self.store["base_dates"][:]

    def base_date(self, index: int) -> np.datetime64:
        """Return the base date at ``index``.

        Parameters
        ----------
        index : int
            Position along the base-date axis.

        Returns
        -------
        np.datetime64
            The base date at ``index``.
        """
        return self.base_dates[index]

    @property
    def base_start_date(self) -> np.datetime64:
        """Return the first base date."""
        return self.base_dates[0]

    @property
    def base_end_date(self) -> np.datetime64:
        """Return the last base date."""
        return self.base_dates[-1]

    @property
    def base_frequency(self) -> datetime.timedelta:
        """Return the interval between consecutive base dates."""
        try:
            return frequency_to_timedelta(self.store.attrs["frequency"])
        except KeyError:
            LOG.warning("No 'frequency' in %r, computing from 'base_dates'", self)
        dates = self.base_dates
        return dates[1].astype(object) - dates[0].astype(object)

    # ------------------------------------------------------------------
    # Step axis
    # ------------------------------------------------------------------

    @cached_property
    def steps(self) -> NDArray[np.timedelta64]:
        """Return the forecast step values stored in the dataset."""
        return self.store["steps"][:]

    @property
    def step_start(self) -> datetime.timedelta:
        """Return the first forecast step."""
        return self.steps[0].astype("timedelta64[s]").astype(datetime.timedelta)

    @property
    def step_end(self) -> datetime.timedelta:
        """Return the last forecast step."""
        return self.steps[-1].astype("timedelta64[s]").astype(datetime.timedelta)

    @property
    def step_frequency(self) -> datetime.timedelta | None:
        """Return the step interval, or None if steps are not uniformly spaced."""
        if len(self.steps) < 2:
            return None
        diffs = np.diff(self.steps)
        if np.all(diffs == diffs[0]):
            return diffs[0].astype("timedelta64[s]").astype(datetime.timedelta)
        return None

    # ------------------------------------------------------------------
    # Envelope (valid-time range)
    # ------------------------------------------------------------------

    @property
    def start_date(self) -> np.datetime64:
        """Return the earliest valid time: first base date + first step."""
        return self.base_dates[0] + self.steps[0]

    @property
    def end_date(self) -> np.datetime64:
        """Return the latest valid time: last base date + last step."""
        return self.base_dates[-1] + self.steps[-1]

    @property
    def latitudes(self) -> NDArray[Any]:
        """Return the latitudes of the grid."""
        try:
            return self.store["latitudes"][:]
        except KeyError:
            LOG.warning("No 'latitudes' in %r, trying 'latitude'", self)
            return self.store["latitude"][:]

    @property
    def longitudes(self) -> NDArray[Any]:
        """Return the longitudes of the grid."""
        try:
            return self.store["longitudes"][:]
        except KeyError:
            LOG.warning("No 'longitudes' in %r, trying 'longitude'", self)
            return self.store["longitude"][:]

    # ------------------------------------------------------------------
    # Shape and type
    # ------------------------------------------------------------------

    @cached_property
    def shape(self) -> Shape:
        """Return the 5-D shape ``(dates, variables, ensembles, steps, cells)``."""
        return self.data.shape

    @cached_property
    def dtype(self) -> np.dtype:
        """Return the data type of the dataset."""
        return self.data.dtype

    @cached_property
    def chunks(self):
        """Return the chunk sizes of the data array."""
        return self.data.chunks

    @property
    def field_shape(self) -> tuple:
        """Return the shape of a single grid field (1-D for flat grids)."""
        try:
            return tuple(self.store.attrs["field_shape"])
        except KeyError:
            return (self.shape[-1],)

    # ------------------------------------------------------------------
    # Variable metadata
    # ------------------------------------------------------------------

    @property
    def name_to_index(self) -> dict[str, int]:
        """Return the mapping of variable names to their axis-1 indices."""
        if "variables" in self.store.attrs:
            return {n: i for i, n in enumerate(self.store.attrs["variables"])}
        return self.store.attrs["name_to_index"]

    @property
    def variables(self) -> list[str]:
        """Return the variable names in axis-1 order."""
        return [k for k, v in sorted(self.name_to_index.items(), key=lambda x: x[1])]

    # ------------------------------------------------------------------
    # Dataset-level metadata
    # ------------------------------------------------------------------

    @property
    def frequency(self) -> datetime.timedelta:
        """Trajectories datasets have two frequencies.

        Use ``base_frequency`` for the base-date interval and
        ``step_frequency`` for the step interval.
        """
        raise AttributeError(
            "Trajectories datasets have two frequencies. "
            "Use 'base_frequency' for the base-date interval and 'step_frequency' for the step interval."
        )

    @property
    def resolution(self) -> str | None:
        """Return the grid resolution string."""
        return self.store.attrs.get("resolution")

    def dataset_metadata(self) -> dict[str, Any]:
        """Return dataset metadata using trajectory-compatible properties."""
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

    @property
    def variables_metadata(self) -> dict[str, Any]:
        """Return per-variable metadata dict."""
        return self.store.attrs.get("variables_metadata", {})

    # ------------------------------------------------------------------
    # Date filtering
    # ------------------------------------------------------------------

    def _frequency_to_indices(self, frequency: str) -> list[int]:
        """Convert a frequency string to base-date indices."""
        from anemoi.utils.dates import frequency_to_seconds

        requested = frequency_to_seconds(frequency)
        dataset = frequency_to_seconds(self.base_frequency)

        if requested % dataset != 0:
            raise ValueError(
                f"Requested frequency {frequency} is not a multiple of the base-date frequency {self.base_frequency}."
            )
        step = requested // dataset
        return range(0, len(self), step)

    def _dates_to_indices(
        self,
        start: None | str | datetime.datetime,
        end: None | str | datetime.datetime,
    ) -> list[int]:
        """Convert a valid-time window to base-date indices using strict envelope filtering.

        A base date is kept if and only if its entire step range falls within
        [start, end]:  ``base + step_start >= start AND base + step_end <= end``.
        """
        from anemoi.datasets.usage.misc import as_first_date
        from anemoi.datasets.usage.misc import as_last_date

        base_dates = self.base_dates
        step_start = self.steps[0]
        step_end = self.steps[-1]

        # Build the envelope dates for snapping: base + step_start / base + step_end
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

        # base + step_start >= start  →  base >= start - step_start
        # base + step_end   <= end    →  base <= end   - step_end
        base_min = start - step_start
        base_max = end - step_end

        return [i for i, d in enumerate(base_dates) if base_min <= d <= base_max]

    # ------------------------------------------------------------------
    # Misc Dataset interface
    # ------------------------------------------------------------------

    def source(self, index: int) -> Source:
        """Return a debug source object for the given date index."""
        return Source(self, index, info=self.path)

    def tree(self) -> Node:
        """Return the tree representation of the dataset."""
        return Node(self, [], path=self.path)

    def get_dataset_names(self, names: set[str]) -> None:
        """Add this dataset's name to the collected set."""
        name, _ = os.path.splitext(os.path.basename(self.path))
        names.add(name)

    def collect_supporting_arrays(self, collected: set, *path: str) -> None:
        """Collect supporting arrays (no-op for leaf stores)."""

    def collect_input_sources(self, collected: set) -> None:
        """Collect input sources (no-op for leaf stores)."""

    @cached_property
    def missing(self) -> set[int]:
        """Return the set of missing base-date indices.

        Empty for the plain :class:`TrajectoriesZarr`; populated by
        :class:`TrajectoriesZarrWithMissingDates` when the store records a
        non-empty ``missing_dates`` attribute.
        """
        return set()

    @cached_property
    def constant_fields(self) -> list[str]:
        """Return the list of variables that are constant across time."""
        result = self.store.attrs.get("constant_fields")
        if result is not None:
            return result
        return self.computed_constant_fields()

    def usage_factory_load(self, name: str) -> Any:
        """Load an operation class from the trajectories sub-package."""
        return self._usage_factory_load(name, __package__)


class TrajectoriesZarrWithMissingDates(TrajectoriesZarr):
    """A trajectories zarr dataset with one or more missing base dates.

    The store records the missing base dates under the ``missing_dates`` zarr
    attribute (set by :class:`anemoi.datasets.create.trajectories.creator.
    TrajectoryGriddedCreator` from the recipe ``base_dates: { missing: ... }``
    list).  Indexing along axis 0 raises :class:`MissingDateError` when a
    missing base date is hit; otherwise the dataset behaves like a regular
    :class:`TrajectoriesZarr`.
    """

    def __init__(self, group: zarr.Group, path: str = None) -> None:
        super().__init__(group, path=path)

        missing = self.store.attrs.get("missing_dates", [])
        missing = {np.datetime64(x, "s") for x in missing}
        self.missing_to_dates = {i: d for i, d in enumerate(self.base_dates) if d in missing}
        self._missing = set(self.missing_to_dates)

    @property
    def missing(self) -> set[int]:
        """Return the set of base-date indices that are missing."""
        return self._missing

    def mutate(self) -> Dataset:
        """Idempotent: already wrapped."""
        return self

    @debug_indexing
    @expand_list_indexing
    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Same as :meth:`TrajectoriesZarr.__getitem__` but raises on missing dates."""
        first = n[0] if isinstance(n, tuple) else n
        if isinstance(first, int):
            hit = first if first in self._missing else None
        elif isinstance(first, slice):
            hit = next(iter(set(range(*first.indices(len(self)))) & self._missing), None)
        elif isinstance(first, (list, tuple)):
            hit = next(iter(set(first) & self._missing), None)
        else:
            raise TypeError(f"Unsupported index {n!r} ({type(first).__name__})")

        if hit is not None:
            self._report_missing(hit)
        return self.data[n]

    def collect_read_parts(self, n):
        if isinstance(n, int):
            if n in self._missing:
                self._report_missing(n)
        elif isinstance(n, slice):
            common = set(range(*n.indices(len(self)))) & self._missing
            if common:
                self._report_missing(next(iter(common)))
        elif isinstance(n, tuple):
            first = n[0]
            if isinstance(first, int):
                if first in self._missing:
                    self._report_missing(first)
            elif isinstance(first, slice):
                common = set(range(*first.indices(len(self)))) & self._missing
                if common:
                    self._report_missing(next(iter(common)))
            elif isinstance(first, (list, tuple)):
                common = set(first) & self._missing
                if common:
                    self._report_missing(next(iter(common)))
        return super().collect_read_parts(n)

    def _report_missing(self, n: int) -> None:
        from anemoi.datasets import MissingDateError

        raise MissingDateError(f"Base date {self.missing_to_dates[n]} is missing (index={n})")

    def tree(self) -> Node:
        return Node(self, [], path=self.path, missing=sorted(self._missing))

    @property
    def label(self) -> str:
        return "trajectories*"
