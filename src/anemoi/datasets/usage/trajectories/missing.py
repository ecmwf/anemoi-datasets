# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Missing-base-date wrapper for trajectories datasets."""

import logging
from typing import Any

from anemoi.datasets import MissingDateError
from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.gridded.missing import MissingDates

LOG = logging.getLogger(__name__)


class MissingBaseDates(MissingDates):
    """Force base dates of a trajectories dataset to be missing.

    Same behaviour as the gridded :class:`MissingDates` wrapper, but the
    indices and date values refer to the ``base_dates`` axis (trajectories
    datasets have no single ``dates`` array).
    """

    def _axis_dates(self, dataset: Dataset) -> Any:
        return dataset.base_dates

    def _report_missing(self, n: int) -> None:
        raise MissingDateError(f"Base date {self._axis_dates(self.forward)[n]} is missing (index={n})")

    @property
    def reason(self) -> dict[str, Any]:
        """Provide the reason for the missing base dates."""
        return {"missing_base_dates": self.missing_dates}

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        return {"missing_base_dates": self.missing_dates}

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        """Override to avoid calling self.frequency (trajectory datasets have two frequencies)."""
        from anemoi.utils.dates import frequency_to_string

        action = self.__class__.__name__.lower()
        return dict(
            action=action,
            variables=self.variables,
            shape=self.shape,
            base_frequency=frequency_to_string(self.base_frequency),
            step_frequency=frequency_to_string(self.step_frequency),
            start_date=str(self.start_date),
            end_date=str(self.end_date),
            base_start_date=str(self.base_start_date),
            base_end_date=str(self.base_end_date),
            forward=self.forward.metadata_specific(),
            **self.forwards_subclass_metadata_specific(),
            **kwargs,
        )

    def dataset_metadata(self) -> dict[str, Any]:
        """Override to avoid calling self.frequency (trajectory datasets have two frequencies)."""
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

    def statistics_tendencies(self, delta: Any = None) -> dict[str, Any]:
        """Delegate to the underlying dataset, which resolves a default delta
        from the step frequency for trajectories.
        """
        return self.forward.statistics_tendencies(delta)
