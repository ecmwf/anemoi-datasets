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

from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.order import build_remapping

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
    # ``output.order_by`` recipe field is validated against this value in
    # ``Recipe.__init__``.
    order_by: list[str] = ["valid_datetime", "param_level", "number"]

    def __init__(self, recipe: Any) -> None:
        """Initialise a SimpleGriddedContext instance.

        Parameters
        ----------
        recipe : Any
            The recipe object containing configuration for output and build steps.
        """

        super().__init__(recipe)

        self.remapping = build_remapping(recipe.build.remapping)

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
        return SimpleGriddedResult(self, argument, data)

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
            raise ValueError(
                f"Gridded layout does not support filtering by {unsupported}. "
                "Use 'dates' instead."
            )

        filtering_dates = filters["dates"]
        return GroupOfDates(sorted(set(group_of_dates) & set(filtering_dates)), group_of_dates.provider)

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
