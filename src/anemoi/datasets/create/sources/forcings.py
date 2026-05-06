# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from anemoi.transform.fields import new_field_with_metadata
from earthkit.data import from_source
from earthkit.data.indexing.fieldlist import SimpleFieldList

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.source import Source

from . import source_registry


@source_registry.register("forcings")
class ForcingsSource(Source):

    def __init__(self, context: Any, template: Any, param: list[str]) -> None:
        super().__init__(context)
        self.template = template
        self.param = param

    def execute_valid_dates(self, dates: ValidDates) -> Any:
        self.context.trace("\u2705", f"from_source(forcings, {self.template}, {self.param}")
        return from_source("forcings", source_or_dataset=self.template, date=list(dates), param=self.param)

    def execute_forecast_dates(self, dates: ForecastDates) -> Any:
        self.context.trace("\u2705", f"from_source(forcings, {self.template}, {self.param}")

        valid_times = [vt for vt, _bt in dates]
        fields = from_source("forcings", source_or_dataset=self.template, date=valid_times, param=self.param)

        # Index forcing fields by valid_datetime for quick lookup
        fields_by_vdt = {}
        for f in fields:
            fields_by_vdt.setdefault(f.metadata("valid_datetime"), []).append(f)

        # For each (valid_time, basetime) pair, clone the forcing fields
        # with the correct date/time/step metadata
        result = []
        for vt, bt in dates:
            step_hours = int((vt - bt).total_seconds() // 3600)
            meta = dict(
                date=int(bt.strftime("%Y%m%d")),
                time=int(bt.strftime("%H%M")),
                step=step_hours,
            )
            for f in fields_by_vdt.get(vt.isoformat(), []):
                result.append(new_field_with_metadata(f, **meta))

        return SimpleFieldList(result)
