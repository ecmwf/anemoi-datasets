# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.usage.forwards import Forwards


class TrajectoryForwards(Forwards):
    """A :class:`Forwards` wrapper over a trajectories dataset.

    Trajectories datasets have two frequencies (``base_frequency`` and
    ``step_frequency``) rather than a single ``frequency``. This mixin injects
    the trajectory-specific keys into the metadata produced by the base
    :class:`Forwards`/:class:`Dataset` machinery, so the concrete wrappers only
    need to provide their own :meth:`forwards_subclass_metadata_specific`.

    It must appear before the gridded parent in the MRO, e.g.
    ``class MissingBaseDates(TrajectoryForwards, MissingDates)``.
    """

    def _trajectory_metadata(self) -> dict[str, Any]:
        """Return the trajectory-specific metadata keys shared by all wrappers."""
        step_frequency = self.step_frequency
        return dict(
            base_frequency=frequency_to_string(self.base_frequency),
            step_frequency=frequency_to_string(step_frequency) if step_frequency is not None else None,
            base_start_date=str(self.base_start_date),
            base_end_date=str(self.base_end_date),
            step_start=frequency_to_string(self.step_start),
            step_end=frequency_to_string(self.step_end),
        )

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        """Return specific metadata, adding the trajectory keys to the base chain."""
        return super().metadata_specific(**self._trajectory_metadata(), **kwargs)

    def dataset_metadata(self) -> dict[str, Any]:
        """Return dataset metadata with the trajectory keys exposed at the top level."""
        md = super().dataset_metadata()
        md.update(self._trajectory_metadata())
        return md
