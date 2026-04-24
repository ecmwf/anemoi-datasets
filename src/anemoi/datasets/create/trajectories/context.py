# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

import numpy as np
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.core.order import build_remapping

from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.trajectories.result import TrajectoryGriddedResult

LOG = logging.getLogger(__name__)


class Steps:
    def __init__(self, steps: dict):
        start = frequency_to_timedelta(steps["start"])
        end = frequency_to_timedelta(steps["end"])
        frequency = frequency_to_timedelta(steps["frequency"])
        self.steps = np.arange(start, end + frequency, frequency)

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __array__(self, dtype=None, copy=None):
        arr = self.steps if dtype is None else self.steps.astype(dtype)
        return arr.copy() if copy else arr


class TrajectoryGriddedContext(Context):
    """Context for building trajectories output data.

    This class extends the base Context to provide additional logic and configuration
    for trajectories datasets, including remapping, grid flattening, and origin tracking.
    """

    # Fixed cube ordering for trajectories: a single composite ``traj_point``
    # axis (one cubelet per ``(basetime, step)`` pair), then variables and
    # ensembles. Not user-configurable on purpose -- this ordering is tightly
    # coupled to the ``traj_point`` remapping key injected below, and
    # per-field placement in ``TrajectoryGriddedCreator.load_result`` reads
    # ``date/time/step`` from field metadata, so cube ordering does not
    # affect the output.
    order_by: list[str] = ["traj_point", "param_level", "number"]

    def __init__(self, recipe: Any) -> None:
        super().__init__(recipe)
        self.steps = Steps(self.recipe.steps)
        # Inject a composite remapping key so that the trajectory axis
        # ``(date, time, step)`` collapses into a single axis in the cube.
        # Without this, ``ds.cube(['date', 'time', 'step', ...])`` would
        # build a full Cartesian product of unique date * unique time *
        # unique step values, which is over-counted whenever the set of
        # (basetime, step) pairs is not a Cartesian product of basetimes
        # and steps -- e.g. base_dates with frequency 12h starting at 00Z
        # but ending at 00Z (an odd number of basetimes).
        remapping = dict(recipe.build.remapping)
        remapping.setdefault("traj_point", "{date}_{time}_{step}")
        self.remapping = build_remapping(remapping)

    def create_result(self, argument: Any, data: Any) -> TrajectoryGriddedResult:
        return TrajectoryGriddedResult(self, argument, data)

    def matching_dates(self, filtering_dates: Any, group_of_dates: Any) -> Any:
        """Find dates that match between filtering_dates and group_of_dates.

        Parameters
        ----------
        filtering_dates : Any
            The dates to filter by.
        group_of_dates : ForecastDates
            The ``(valid_time, basetime)`` pairs for this group.

        Returns
        -------
        ForecastDates
            A ForecastDates containing only the pairs whose basetime is in
            ``filtering_dates``.
        """
        from anemoi.datasets.create.arguments import ForecastDates

        filter_set = set(filtering_dates)
        matched = [(vt, bt) for vt, bt in group_of_dates if bt in filter_set]
        return ForecastDates(matched)

    def origin(self, data: Any, action: Any, action_arguments: Any) -> Any:
        """Update the origin metadata for each field in the data.

        Parameters
        ----------
        data : Any
            The data fields to update.
        action : Any
            The action providing the new origin.
        action_arguments : Any
            Arguments for the action.

        Returns
        -------
        Any
            A new field list with updated origin metadata.
        """

        origin = action.origin()

        result = []
        for fs in data:
            previous = fs.metadata("anemoi_origin", default=None)
            fall_through = fs.metadata("anemoi_fall_through", default=False)
            if fall_through:
                # The field has pass unchanges in a filter
                result.append(fs)
            else:
                anemoi_origin = origin.combine(previous, action, action_arguments)
                result.append(new_field_with_metadata(fs, anemoi_origin=anemoi_origin))

        return new_fieldlist_from_list(result)
