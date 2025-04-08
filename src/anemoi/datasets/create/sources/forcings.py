# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import List

from earthkit.data import from_source

from .legacy import legacy_source


@legacy_source(__file__)
def forcings(context: Any, dates: List[str], template: str, param: str) -> Any:
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
    context.trace("✅", f"from_source(forcings, {template}, {param}")
    return from_source("forcings", source_or_dataset=template, date=dates, param=param)


execute = forcings
