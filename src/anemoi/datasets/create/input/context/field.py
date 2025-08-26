# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any

from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data.core.order import build_remapping

from ..result.field import FieldResult
from . import Context


class FieldContext(Context):

    def __init__(
        self,
        /,
        argument: Any,
        order_by: str,
        flatten_grid: bool,
        remapping: dict[str, Any],
        use_grib_paramid: bool,
    ) -> None:
        super().__init__(argument)
        self.order_by = order_by
        self.flatten_grid = flatten_grid
        self.remapping = build_remapping(remapping)
        self.use_grib_paramid = use_grib_paramid
        self.partial_ok = False

    def empty_result(self) -> Any:
        import earthkit.data as ekd

        return ekd.from_source("empty")

    def source_argument(self, argument: Any) -> Any:
        return argument  # .dates

    def filter_argument(self, argument: Any) -> Any:
        return argument

    def create_result(self, data):
        return FieldResult(self, data)

    def matching_dates(self, filtering_dates, group_of_dates: Any) -> Any:
        from anemoi.datasets.dates.groups import GroupOfDates

        return GroupOfDates(sorted(set(group_of_dates) & set(filtering_dates)), group_of_dates.provider)

    def origin(self, data: Any, action: Any) -> Any:

        origin = action.origin()

        result = []
        for fs in data:
            previous = fs.metadata("anemoi_origin", default=None)
            origin = origin.combine(previous)
            result.append(new_field_with_metadata(fs, anemoi_origin=origin))

        result = new_fieldlist_from_list(result)

        for fs in result:
            fs.metadata()

        return result
