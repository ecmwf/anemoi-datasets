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

from anemoi.datasets.create.gridded.result import SimpleGriddedResult
from anemoi.datasets.create.input.context import Context
from anemoi.datasets.dates.groups import GroupOfDates

LOG = logging.getLogger(__name__)


class SimpleGriddedContext(Context):
    """Context for building gridded output data.

    This class extends the base Context to provide additional logic and configuration
    for gridded datasets, including remapping, grid flattening, and origin tracking.
    """

    # Fixed cube ordering for gridded datasets. Not user-configurable: the
    # last two keys are assumed to be ``(variables, ensembles)`` by
    # ``BaseResult.build_coords``, and the first key is the time axis --
    # varying any of this breaks the coord construction. The deprecated
    # ``output.order_by`` recipe field is validated against the old bare-key
    # form (["valid_datetime", "param_level", "number"]) in ``Recipe.__init__``.
    # Use ``time.valid_datetime`` (the time component path) rather than
    # ``metadata.valid_datetime`` (raw GRIB metadata path): the time-component
    # path survives field.set() wrapping (e.g. new_field_with_metadata) whereas
    # the raw-metadata path does not. ``labels.name`` is the field name
    # attached by the naming scheme (see ``anemoi.transform.naming``).
    order_by: list[str] = ["time.valid_datetime", "labels.name", "metadata.number"]

    def __init__(self, recipe: Any) -> None:
        """Initialise a SimpleGriddedContext instance.

        Parameters
        ----------
        recipe : Any
            The recipe object containing configuration for output and build steps.
        """

        super().__init__(recipe)

    def create_result(self, argument: Any, data: Any) -> SimpleGriddedResult:
        """Create a SimpleGriddedResult object for the given argument and data.

        Parameters
        ----------
        argument : Any
            The argument used to create the result.
        data : Any
            The data to be wrapped in the result.

        Returns
        -------
        SimpleGriddedResult
            The created SimpleGriddedResult instance.
        """
        return SimpleGriddedResult(self, argument, self.naming(data))

    def matching_dates(self, filters: dict, group_of_dates: Any) -> GroupOfDates:
        """Find dates that match between filters and group_of_dates.

        Parameters
        ----------
        filters : dict
            A dict mapping filter keys to DatesProvider objects.
            Gridded layouts only support ``'dates'``.
        group_of_dates : Any
            The group of dates to compare against.

        Returns
        -------
        GroupOfDates
            A GroupOfDates object containing the intersection of the two sets.
        """
        unsupported = set(filters) - {"dates"}
        if unsupported:
            raise ValueError(f"Gridded layout does not support filtering by {unsupported}. " "Use 'dates' instead.")

        filtering_dates = filters["dates"]
        return GroupOfDates(sorted(set(group_of_dates) & set(filtering_dates)), group_of_dates.provider)
