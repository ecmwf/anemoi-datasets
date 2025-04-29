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

import earthkit.data as ekd

from .legacy import legacy_source
from .xarray import load_many


@legacy_source(__file__)
def execute(context: Any, dates: List[str], url: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
    """Execute the data loading process.

    Parameters
    ----------
    context : Any
        The context in which the execution occurs.
    dates : List[str]
        List of dates for which data is to be loaded.
    url : str
        The URL from which data is to be loaded.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        The loaded data.
    """
    return load_many("🇿", context, dates, url, *args, **kwargs)
