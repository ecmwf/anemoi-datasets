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


@source_registry.register("forcings")
class ForcingsSource(LegacySource):

    @staticmethod
    def _execute(context: Any, dates: list[str], template: str, param: str) -> Any:
        """Loads forcing data from a specified source.

    Returns
    -------
    object
        Loaded forcing data.
    """
    context.trace("✅", f"from_source(forcings, {template}, {param}")
    return from_source("forcings", source_or_dataset=template, date=list(dates), param=param)

        Returns
        -------
        object
            Loaded forcing data.
        """
        context.trace("✅", f"from_source(forcings, {template}, {param}")
        return from_source("forcings", source_or_dataset=template, date=list(dates), param=param)
