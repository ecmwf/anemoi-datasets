# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import warnings
from typing import Any

from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list

from ..result.observations import ObservationsResult
from . import Context

LOG = logging.getLogger(__name__)


class ObservationContext(Context):

    def __init__(
        self,
        /,
        argument: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(argument)

    def empty_result(self) -> Any:
        return []

    def source_argument(self, argument: Any) -> Any:
        return argument  # .dates

    def filter_argument(self, argument: Any) -> Any:
        return argument

    def create_result(self, data):
        return ObservationsResult(self, data)

    def matching_dates(self, filtering_dates, group_of_dates: Any) -> Any:
        from anemoi.datasets.dates.groups import GroupOfDates

        return GroupOfDates(sorted(set(group_of_dates) & set(filtering_dates)), group_of_dates.provider)

    def origin(self, data: Any, action: Any, action_arguments: Any) -> Any:
        warnings.warn("ObservationContext.origin is not implemented", UserWarning)
        return data

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
