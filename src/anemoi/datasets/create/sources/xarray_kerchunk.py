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
from typing import Optional

import earthkit.data as ekd
from earthkit.data.core.fieldlist import MultiFieldList

from .patterns import iterate_patterns
from .xarray import load_one


def load_many(
    emoji: str, context: Any, dates: List[str], pattern: str, options: Optional[Dict[str, Any]], **kwargs: Any
) -> ekd.FieldList:
    """Loads multiple datasets based on the provided pattern and dates.

    Parameters
    ----------
    emoji : str
        An emoji representing the dataset type.
    context : object
        The context in which the datasets are loaded.
    dates : list
        List of dates for which the datasets are to be loaded.
    pattern : str
        The pattern to match the dataset paths.
    options : dict, optional
        Additional options for loading the datasets.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ekd.FieldList
        A list of loaded datasets.
    """

    result = []
    options = options.copy() if options is not None else {}

    options.setdefault("engine", "zarr")
    options.setdefault("backend_kwargs", {})

    backend_kwargs = options["backend_kwargs"]
    backend_kwargs.setdefault("consolidated", False)
    backend_kwargs.setdefault("storage_options", {})

    storage_options = backend_kwargs["storage_options"]
    storage_options.setdefault("remote_protocol", "s3")
    storage_options.setdefault("remote_options", {"anon": True})

    for path, dates in iterate_patterns(pattern, dates, **kwargs):
        storage_options["fo"] = path

        result.append(load_one(emoji, context, dates, "reference://", options=options, **kwargs))

    return MultiFieldList(result)


def execute(
    context: Any, dates: List[str], json: str, options: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> ekd.FieldList:
    """Executes the loading of datasets using the provided context and dates.

    Parameters
    ----------
    context : object
        The context in which the datasets are loaded.
    dates : list
        List of dates for which the datasets are to be loaded.
    json : str
        The JSON pattern to match the dataset paths.
    options : dict, optional
        Additional options for loading the datasets.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    MultiFieldList
        A list of loaded datasets.
    """
    return load_many("🧱", context, dates, json, options, **kwargs)
