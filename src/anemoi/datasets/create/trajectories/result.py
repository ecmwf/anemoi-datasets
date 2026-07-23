# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import cached_property
from typing import Any

from anemoi.datasets.create.gridded.result import GriddedResult


class TrajectoryGriddedResult(GriddedResult):
    """Result class for trajectory-indexed gridded datasets.

    Extends ``GriddedResult`` with a ``steps`` coordinate taken from the
    trajectories context (which mirrors ``recipe.steps``).  The cube itself is
    ordered by ``valid_datetime`` first (see ``TrajectoriesOutput.order_by``);
    the ``(basetime, step)`` pair is recovered per-field in the creator's
    ``load_result`` from the field metadata.
    """

    @property
    def steps(self) -> Any:
        """Retrieve the step values for the result (from context)."""
        return self.context.steps

    @cached_property
    def coords(self) -> dict[str, Any]:
        """Retrieve the coordinates of the result."""
        self.build_coords()
        return {
            "dates": self._basetimes(),
            "steps": self.steps,
            "variables": self.variables,
            "ensembles": self.ensembles,
            "values": self.grid_values,
        }

    def _basetimes(self) -> list:
        """Return the distinct basetimes in the argument, in order."""
        seen: list = []
        seen_set: set = set()
        for _valid, basetime in self.group_of_dates.items:
            if basetime not in seen_set:
                seen_set.add(basetime)
                seen.append(basetime)
        return seen
