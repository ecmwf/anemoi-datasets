# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Dict

from earthkit.data.core.order import build_remapping

from . import Context
from ..result.field import FieldResult


class FieldContext(Context):

    def __init__(self, /, order_by: str, flatten_grid: bool, remapping: Dict[str, Any], use_grib_paramid: bool) -> None:
        super().__init__()
        self.order_by = order_by
        self.flatten_grid = flatten_grid
        self.remapping = build_remapping(remapping)
        self.use_grib_paramid = use_grib_paramid

    def empty_result(self) -> Any:
        import earthkit.data as ekd

        return ekd.from_source("empty")

    def source_argument(self, argument: Any) -> Any:
        return argument.dates

    def create_result(self, data):
        return FieldResult(self, data)
