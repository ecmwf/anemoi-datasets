# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

from anemoi.transform.fields import build_remapping
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list

from anemoi.datasets.create.input.context import Context
from anemoi.datasets.create.recipe.dates import Steps  # noqa: F401  (re-exported for back-compat)
from anemoi.datasets.create.trajectories.result import TrajectoryGriddedResult

LOG = logging.getLogger(__name__)


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
    # affect the output. Keys use the ``metadata.`` prefix required by
    # earthkit 1.0's ``_get_single``; ``traj_point`` and ``param_level`` are
    # remapped synthetic keys and have no prefix.
    order_by: list[str] = ["traj_point", "param_level", "metadata.number"]

    def __init__(self, recipe: Any) -> None:
        super().__init__(recipe)
        self.steps = self.recipe.steps
        # Inject a composite remapping key so that the trajectory axis
        # ``(date, time, step)`` collapses into a single axis in the cube.
        # Without this, ``ds.cube(['date', 'time', 'step', ...])`` would
        # build a full Cartesian product of unique date * unique time *
        # unique step values, which is over-counted whenever the set of
        # (basetime, step) pairs is not a Cartesian product of basetimes
        # and steps -- e.g. base_dates with frequency 12h starting at 00Z
        # but ending at 00Z (an odd number of basetimes).
        remapping = dict(recipe.build.remapping)
        remapping.setdefault("traj_point", "{time.base_datetime}_{time.step}")
        self.remapping = build_remapping(remapping)

    def create_result(self, argument: Any, data: Any) -> TrajectoryGriddedResult:
        return TrajectoryGriddedResult(self, argument, data)

    def matching_dates(self, filters: dict, group_of_dates: Any) -> Any:
        """Find dates that match between filters and group_of_dates.

        Parameters
        ----------
        filters : dict
            A dict mapping filter keys to DatesProvider objects.
            Trajectory layouts support ``'base_dates'`` and ``'steps'``.
        group_of_dates : ForecastDates
            The ``(valid_time, basetime)`` pairs for this group.

        Returns
        -------
        ForecastDates
            A ForecastDates containing only the matching pairs.
        """
        unsupported = set(filters) - {"base_dates", "steps"}
        if unsupported:
            raise ValueError(
                f"Trajectory layout does not support filtering by {unsupported}. "
                "Use 'base_dates' and/or 'steps' instead."
            )

        from anemoi.datasets.create.arguments import ForecastDates

        matched = list(group_of_dates)

        if "base_dates" in filters:
            base_dates_set = set(filters["base_dates"])
            matched = [(vt, bt) for vt, bt in matched if bt in base_dates_set]

        if "steps" in filters:
            steps_set = set(filters["steps"])
            matched = [(vt, bt) for vt, bt in matched if (vt - bt) in steps_set]

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
            previous = fs.get("metadata.anemoi_origin", default=None)
            fall_through = fs.get("metadata.anemoi_fall_through", default=False)
            if fall_through:
                # The field has pass unchanges in a filter
                result.append(fs)
            else:
                anemoi_origin = origin.combine(previous, action, action_arguments)
                result.append(new_field_with_metadata(fs, anemoi_origin=anemoi_origin))

        return new_fieldlist_from_list(result)
