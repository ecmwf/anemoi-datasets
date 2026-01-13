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

import earthkit.data as ekd
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.order import build_remapping

from anemoi.datasets.create.gridded.result import GriddedResult
from anemoi.datasets.create.input.context import Context
from anemoi.datasets.dates.groups import GroupOfDates

LOG = logging.getLogger(__name__)


class GriddedContext(Context):
    """Context for building gridded output data.

    This class extends the base Context to provide additional logic and configuration
    for gridded datasets, including remapping, grid flattening, and origin tracking.
    """

    def __init__(self, recipe: Any) -> None:
        """Initialise a GriddedContext instance.

        Parameters
        ----------
        recipe : Any
            The recipe object containing configuration for output and build steps.
        """

        super().__init__(recipe)

        self.order_by = recipe.output.order_by
        self.flatten_grid = recipe.output.flatten_grid
        self.remapping = build_remapping(recipe.build.remapping)
        self.use_grib_paramid = recipe.build.use_grib_paramid

    def empty_result(self) -> Any:
        """Create an empty result using earthkit.data.

        Returns
        -------
        Any
            An empty data source from earthkit.data.
        """

        return ekd.from_source("empty")

    def create_result(self, argument: Any, data: Any) -> GriddedResult:
        """Create a GriddedResult object for the given argument and data.

        Parameters
        ----------
        argument : Any
            The argument used to create the result.
        data : Any
            The data to be wrapped in the result.

        Returns
        -------
        GriddedResult
            The created GriddedResult instance.
        """
        return GriddedResult(self, argument, data)

    def matching_dates(self, filtering_dates: Any, group_of_dates: Any) -> GroupOfDates:
        """Find dates that match between filtering_dates and group_of_dates.

        Parameters
        ----------
        filtering_dates : Any
            The dates to filter by.
        group_of_dates : Any
            The group of dates to compare against.

        Returns
        -------
        GroupOfDates
            A GroupOfDates object containing the intersection of the two sets.
        """

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
