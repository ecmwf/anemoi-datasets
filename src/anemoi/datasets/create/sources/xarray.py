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
from typing import Optional

import earthkit.data as ekd

from anemoi.datasets.create.typing import DateList

from ..source import Source
from .xarray_support import XarrayFieldList
from .xarray_support import load_many
from .xarray_support import load_one

__all__ = ["load_many", "load_one", "XarrayFieldList"]


class XarraySource(Source):
    """An Xarray base data source, intended to be subclassed."""

    emoji = "✖️"  # For tracing

    options: Optional[Dict[str, Any]] = None
    flavour: Optional[Dict[str, Any]] = None
    patch: Optional[Dict[str, Any]] = None

    def __init__(self, context: Any, **kwargs: dict):
        """Initialise the source.

        Parameters
        ----------
        context : Any
            The context for the data source.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, **kwargs)
        self.kwargs = kwargs

    def execute(self, dates: DateList) -> ekd.FieldList:
        """Execute the data loading process for the given dates.

        Parameters
        ----------
        dates : DateList
            List of dates for which data needs to be loaded.

        Returns
        -------
        ekd.FieldList
            The loaded data fields.
        """

        # For now, just a simple wrapper around load_many
        # TODO: move the implementation here

        return load_many(
            self.emoji,
            self.context,
            dates,
            pattern=self.path_or_url,
            options=self.options,
            flavour=self.flavour,
            patch=self.patch,
            **self.kwargs,
        )
