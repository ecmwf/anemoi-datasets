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

from .xarray import load_many


def execute(context: Dict[str, Any], dates: List[str], url: str, *args: Any, **kwargs: Any) -> Any:
    """Execute the loading of data using xarray and zarr.

    Parameters
    ----------
    context : dict
        The context dictionary containing configuration and state.
    dates : list of str
        The list of dates for which to load data.
    url : str
        The URL from which to load the data.
    *args : Any
        Additional positional arguments.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    Any
        The loaded data.
    """
    return load_many("🇿", context, dates, url, *args, **kwargs)
