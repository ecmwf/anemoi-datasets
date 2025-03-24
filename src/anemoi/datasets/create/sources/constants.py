# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Dict
from typing import List

from earthkit.data import from_source

from .legacy import legacy_source


@legacy_source(__file__)
def constants(context: Any, dates: List[str], template: Dict[str, Any], param: str) -> Any:
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
    context.trace("✅", f"from_source(constants, {template}, {param}")
    if len(template) == 0:
        raise ValueError("Forcings template is empty.")

    return from_source("forcings", source_or_dataset=template, date=dates, param=param)


execute: Any = constants
