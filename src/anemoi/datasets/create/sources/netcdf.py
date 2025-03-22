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
def execute(context: Any, dates: List[str], path: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
    """Execute the loading of multiple NetCDF files.

    Parameters
    ----------
    context : object
        The context in which the function is executed.
    dates : list
        List of dates for which data is to be loaded.
    path : str
        Path to the directory containing the NetCDF files.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    object
        The loaded data.
    """
    return load_many("📁", context, dates, path, *args, **kwargs)
