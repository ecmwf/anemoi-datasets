# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import datetime
from typing import Any

from earthkit.data import from_source

from anemoi.datasets.create.sources import source_registry

from .legacy import LegacySource


@source_registry.register("source")
class GenericSource(LegacySource):

    @staticmethod
    def _execute(context: Any | None, dates: list[datetime], **kwargs: Any) -> Any:
        """Generates a source based on the provided context, dates, and additional keyword arguments.

        Parameters
        ----------
        context : Optional[Any]
            The context in which the source is generated.
        dates : List[datetime]
            A list of datetime objects representing the dates.
        **kwargs : Any
            Additional keyword arguments for the source generation.

        Returns
        -------
        Any
            The generated source.
        """
        name = kwargs.pop("name")
        context.trace("✅", f"from_source({name}, {dates}, {kwargs}")
        if kwargs["date"] == "$from_dates":
            kwargs["date"] = list({d.strftime("%Y%m%d") for d in dates})
        if kwargs["time"] == "$from_dates":
            kwargs["time"] = list({d.strftime("%H%M") for d in dates})
        return from_source(name, **kwargs)
