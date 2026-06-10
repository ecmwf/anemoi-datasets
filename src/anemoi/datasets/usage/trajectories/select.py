# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from typing import Any

from anemoi.utils.dates import frequency_to_string
from numpy.typing import NDArray

from ..gridded.select import Select as GriddedSelect

LOG = logging.getLogger(__name__)


class Select(GriddedSelect):
    """Select a subset of variables from a trajectories dataset.

    The variable axis is at position 1 in the 5-D layout
    ``(dates, variables, ensembles, steps, cells)`` — the same position as
    in the gridded 4-D layout, and the gridded indexing machinery
    (``index_to_slices`` etc.) is dimension-agnostic, so the whole data
    path is inherited from the gridded ``Select``.  Only the metadata
    methods are overridden, because trajectory datasets have two
    frequencies and no single ``frequency`` property.
    """

    def statistics_tendencies(self, delta: datetime.timedelta | None = None) -> dict[str, NDArray[Any]]:
        """Return the statistical tendencies, subsetted to the selected variables.

        Parameters
        ----------
        delta : datetime.timedelta, optional
            The time delta for the tendencies.  Defaults to the step
            frequency (the axis along which trajectory tendencies are
            computed at creation time).

        Returns
        -------
        dict of str to NDArray
            The statistical tendencies for the selected variables.
        """
        if delta is None:
            delta = self.step_frequency
            if delta is None:
                raise ValueError("statistics_tendencies: steps are not uniformly spaced, pass `delta` explicitly.")
        return {k: v[self.indices] for k, v in self.dataset.statistics_tendencies(delta).items()}

    def metadata_specific(self, **kwargs: Any) -> dict[str, Any]:
        """Return this wrapper's node of the recursive construction tree.

        Override of the shared implementation only to avoid resolving
        ``self.frequency`` (trajectory datasets have two frequencies:
        ``base_frequency`` and ``step_frequency``).
        """
        action = self.__class__.__name__.lower()
        step_freq = self.step_frequency
        return dict(
            action=action,
            variables=self.variables,
            shape=self.shape,
            base_frequency=frequency_to_string(self.base_frequency),
            step_frequency=frequency_to_string(step_freq) if step_freq is not None else None,
            start_date=str(self.start_date),
            end_date=str(self.end_date),
            base_start_date=str(self.base_start_date),
            base_end_date=str(self.base_end_date),
            forward=self.forward.metadata_specific(),
            **self.forwards_subclass_metadata_specific(),
            **kwargs,
        )

    def dataset_metadata(self) -> dict[str, Any]:
        """Return the flat summary of the resulting dataset.

        Override of the shared implementation only to avoid resolving
        ``self.frequency`` (trajectory datasets have two frequencies:
        ``base_frequency`` and ``step_frequency``).
        """
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
