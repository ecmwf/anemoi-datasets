# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from earthkit.data import from_source

from . import source_registry
from .legacy import LegacySource

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


@source_registry.register("forcings")
class ForcingsSource(LegacySource):

    @staticmethod
    def _execute(context: Any, dates: list[str], template: str, param: str) -> Any:
        """Loads forcing data from a specified source.

        Parameters
        ----------
        context : object
            The context in which the function is executed.
        dates : list
            List of dates for which data is to be loaded.
        template : FieldList
            Template for the data source.
        param : str
            Parameter for the data source.

        Returns
        -------
        object
            Loaded forcing data.
        """
        from anemoi.transform.fields import new_field_with_metadata
        from anemoi.transform.fields import new_fieldlist_from_list

        context.trace("✅", f"from_source(forcings, {template}, {param}")
        fields = from_source("forcings", source_or_dataset=template, date=list(dates), param=param)
        result = []
        for field in fields:
            units = field.metadata("units", default=None)
            if units is not None:
                assert False, units
                result.append(field)
            else:
                name = field.metadata("param")
                result.append(new_field_with_metadata(field, units=UNITS[name]))

        return new_fieldlist_from_list(result)


@source_registry.register("constants")
class ConstantsSource(LegacySource):

    @staticmethod
    def _execute(context: Any, dates: list[str], template: dict[str, Any], param: str) -> Any:
        """Deprecated function to retrieve constants data.

        Parameters
        ----------
        context : Any
            The context object for tracing.
        dates : list of str
            List of dates for which data is required.
        template : dict of str to Any
            Template dictionary for the data source.
        param : str
            Parameter to retrieve.

        Returns
        -------
        Any
            Data retrieved from the source.
        """
        from warnings import warn

        warn(
            "The source `constants` is deprecated, use `forcings` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return ForcingsSource._execute(context, dates, template, param)
