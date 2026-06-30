# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Any

from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from earthkit.data import from_source

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.source import Source

from . import source_registry

# This table is not complete and needs updating

UNITS = dict(
    cos_julian_day="dimensionless",
    cos_latitude="dimensionless",
    cos_local_time="dimensionless",
    cos_longitude="dimensionless",
    sin_julian_day="dimensionless",
    sin_latitude="dimensionless",
    sin_local_time="dimensionless",
    sin_longitude="dimensionless",
    cos_solar_zenith_angle="dimensionless",
    insolation="dimensionless",  # An alias for the one above
)


def _units_for(field: Any) -> str:
    """Return the units of a forcing field.

    Looks up the ``UNITS`` table by ``parameter.variable`` first so that
    forcing fields (e.g. cos_latitude) get the correct units even when the
    underlying template GRIB carries a different ``metadata.units`` value.
    """
    param_var = field.get("parameter.variable", default=None)
    if param_var is not None and param_var in UNITS:
        return UNITS[param_var]
    units = field.get("metadata.units", default=None)
    if units is None:
        units = UNITS.get(field.metadata("param"))
    return units


@source_registry.register("forcings")
class ForcingsSource(Source):

    def __init__(self, context: Any, template: Any, param: list[str]) -> None:
        super().__init__(context)
        self.template = template
        self.param = param

    def execute_valid_dates(self, dates: ValidDates) -> Any:
        self.context.trace("\u2705", f"from_source(forcings, {self.template}, {self.param}")
        fields = from_source("forcings", source_or_dataset=self.template, date=list(dates), param=self.param).to_fieldlist()
        result = [new_field_with_metadata(f, units=_units_for(f)) for f in fields]
        return new_fieldlist_from_list(result)

    def execute_forecast_dates(self, dates: ForecastDates) -> Any:
        self.context.trace("\u2705", f"from_source(forcings, {self.template}, {self.param}")

        valid_times = [vt for vt, _bt in dates]
        fields = from_source("forcings", source_or_dataset=self.template, date=valid_times, param=self.param).to_fieldlist()

        # Index forcing fields by valid_datetime for quick lookup
        fields_by_vdt = {}
        for f in fields:
            fields_by_vdt.setdefault(f.time.valid_datetime(), []).append(f)

        # For each (valid_time, basetime) pair, clone the forcing fields
        # with the correct base_datetime/step metadata
        result = []
        for vt, bt in dates:
            step_hours = int((vt - bt).total_seconds() // 3600)
            meta = dict(
                base_datetime=bt,
                step=datetime.timedelta(hours=step_hours),
            )
            for f in fields_by_vdt.get(vt, []):
                result.append(new_field_with_metadata(f, units=_units_for(f), **meta))

        return new_fieldlist_from_list(result)
