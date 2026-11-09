# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Overlay a mix of trajectory and gridded datasets along the variable axis.

``open_dataset(traj, gridded, ...)`` (in any order, any number of each) is
routed here whenever at least one argument is a trajectories dataset.  Each
gridded dataset is broadcast onto the trajectory layout with
:class:`GriddedAsTrajectory` (valid-time lookup), and all datasets are then
joined along the variable axis, preserving the argument order.  The result
behaves like a trajectories dataset.
"""

import datetime
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.gridded.join import Join
from anemoi.datasets.usage.options import Options
from anemoi.datasets.usage.trajectories.broadcast import GriddedAsTrajectory
from anemoi.datasets.usage.trajectories.metadata import trajectory_metadata

LOG = logging.getLogger(__name__)


def is_trajectory(dataset: Dataset) -> bool:
    """Return whether a dataset uses the 5-D trajectories layout.

    Parameters
    ----------
    dataset : Dataset
        The dataset to test.

    Returns
    -------
    bool
        True if the dataset has the 5-D trajectories layout.
    """
    return len(dataset.shape) == 5


class TrajectoryJoin(Join):
    """Join trajectory (and broadcast-gridded) datasets along the variable axis.

    Reuses the whole gridded :class:`~anemoi.datasets.usage.gridded.join.Join`
    data path (variable-axis concatenation is dimension-agnostic) and only
    swaps the compatibility checks — which must compare ``base_dates`` and
    ``steps`` rather than the single ``dates`` axis that trajectories do not
    have — and the metadata, which follows the trajectory convention.

    Parameters
    ----------
    datasets : list of Dataset
        The datasets to join, in argument order.  Gridded members must already
        be wrapped as :class:`GriddedAsTrajectory`.
    options : Options
        Options for the combined dataset.
    """

    def __init__(self, datasets: list[Dataset], options: Options) -> None:
        self._template = next((d for d in datasets if not isinstance(d, GriddedAsTrajectory)), None)
        if self._template is None:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one trajectory member "
                "(all members are broadcast-gridded)."
            )
        super().__init__(datasets, options)

    # ------------------------------------------------------------------
    # Compatibility (base_dates / steps instead of dates)
    # ------------------------------------------------------------------

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        """Check that two members share grid, base dates and steps."""
        # Sub-shapes first: a clean "Incompatible shapes" message for a
        # cell/ensemble/step-count mismatch, before check_same_grid compares
        # (mismatched-length) coordinate arrays.
        self.check_same_sub_shapes(d1, d2, drop_axis=1)
        self.check_same_base_dates(d1, d2)
        self.check_same_steps(d1, d2)
        self.check_same_grid(d1, d2)

    def check_same_base_dates(self, d1: Dataset, d2: Dataset) -> None:
        """Raise if two members have different base dates."""
        if not np.array_equal(d1.base_dates, d2.base_dates):
            raise ValueError(
                f"{self.__class__.__name__}: Incompatible base dates: "
                f"{d1.base_dates[0]}..{d1.base_dates[-1]} and {d2.base_dates[0]}..{d2.base_dates[-1]} ({d1} {d2})"
            )

    def check_same_steps(self, d1: Dataset, d2: Dataset) -> None:
        """Raise if two members have different steps."""
        if not np.array_equal(d1.steps, d2.steps):
            raise ValueError(f"{self.__class__.__name__}: Incompatible steps: {d1.steps} and {d2.steps} ({d1} {d2})")

    # ------------------------------------------------------------------
    # Operation dispatch and date filtering (delegated to a real trajectory)
    # ------------------------------------------------------------------

    def usage_factory_load(self, name: str) -> Any:
        """Resolve operations from the trajectories package via a real trajectory member."""
        return self._template.usage_factory_load(name)

    def _dates_to_indices(self, start: Any, end: Any) -> list[int]:
        """Use the trajectory envelope filtering of the template member.

        The template shares the join's base dates and steps, so its base-date
        indices apply unchanged to the join.
        """
        return self._template._dates_to_indices(start, end)

    def _frequency_to_indices(self, frequency: str) -> list[int]:
        """The ``frequency`` option is not supported for trajectories (delegated)."""
        return self._template._frequency_to_indices(frequency)

    # ------------------------------------------------------------------
    # Metadata (trajectory convention)
    # ------------------------------------------------------------------

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
        """Return joined statistics tendencies, defaulting the delta to the step frequency."""
        if delta is None:
            delta = self.step_frequency
            if delta is None:
                raise ValueError(
                    "Cannot use a default tendencies delta: the steps of this dataset "
                    "are not uniformly spaced. Pass 'delta' explicitly."
                )
        return super().statistics_tendencies(delta)

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        return super().metadata_specific(**trajectory_metadata(self), **kwargs)

    def dataset_metadata(self) -> dict[str, Any]:
        md = super().dataset_metadata()
        md.update(trajectory_metadata(self))
        return md


def trajectory_join(datasets: list[Dataset], options: Options) -> Dataset:
    """Build a :class:`TrajectoryJoin` from a mix of trajectory and gridded datasets.

    Parameters
    ----------
    datasets : list of Dataset
        The opened datasets, in argument order.  At least one must be a
        trajectories dataset.
    options : Options
        Options for the combined dataset.

    Returns
    -------
    Dataset
        The joined dataset, behaving like a trajectories dataset.
    """
    template = next((d for d in datasets if is_trajectory(d)), None)
    if template is None:
        raise ValueError("trajectory_join requires at least one trajectories dataset")

    base_dates = template.base_dates
    steps = template.steps

    members = [d if is_trajectory(d) else GriddedAsTrajectory(d, base_dates, steps) for d in datasets]

    return TrajectoryJoin(members, options)
