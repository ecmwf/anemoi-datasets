# (C) Copyright 2025 Anemoi contributors.
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

from .xarray import load_many


def execute(context: Any, dates: List[str], path: str, *args: Any, **kwargs: Any) -> ekd.FieldList:
    """Execute the loading of data using the specified context, dates, and path.

    Parameters
    ----------
    context : object
        The context in which the data loading is executed.
    dates : list
        List of dates for which data is to be loaded.
    path : str
        The path to the data source.
    *args : tuple
        Additional positional arguments.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    xarray.Dataset
        The loaded dataset.
    """
    options = {"engine": "fstd"}
    return load_many("🍁", context, dates, path, *args, options=options, **kwargs)
