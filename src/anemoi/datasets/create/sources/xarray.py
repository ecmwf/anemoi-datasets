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

    path_is_pattern: bool = True

    def __init__(self, context: Any, *args: tuple, **kwargs: dict):
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
        super().__init__(context, *args, **kwargs)

    def execute(self, dates: DateList) -> ekd.FieldList:
        # For now, just a simple wrapper around load_many
        # TODO: move the implementation here

        if self.pattern:
            return load_many(self.emoji, self.context, dates, pattern=self.pattern)

        return load_one(self.emoji, self.context, dates, self.path, self.options)
