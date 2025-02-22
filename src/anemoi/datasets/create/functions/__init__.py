# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
from typing import Any
from typing import Callable

import entrypoints


def assert_is_fieldlist(obj: Any) -> None:
    """Asserts that the given object is an instance of FieldList.

    Parameters
    ----------
    obj : Any
        The object to check.

    Raises
    ------
    AssertionError
        If the object is not an instance of FieldList.
    """
    from earthkit.data.indexing.fieldlist import FieldList

    assert isinstance(obj, FieldList), type(obj)


def import_function(name: str, kind: str) -> Callable:
    """Imports a function based on the given name and kind.

    Parameters
    ----------
    name : str
        The name of the function to import.
    kind : str
        The kind of function to import (e.g., 'filters', 'sources').

    Returns
    -------
    Callable
        The imported function.

    Raises
    ------
    ValueError
        If the function cannot be found.
    """
    from anemoi.transform.filters import filter_registry
    from anemoi.transform.sources import source_registry

    name = name.replace("-", "_")

    plugins = {}
    for e in entrypoints.get_group_all(f"anemoi.datasets.{kind}"):
        plugins[e.name.replace("_", "-")] = e

    if name in plugins:
        return plugins[name].load()

    try:
        module = importlib.import_module(
            f".{kind}.{name}",
            package=__name__,
        )
        return module.execute
    except ModuleNotFoundError:
        pass

    if kind == "filters":
        if filter_registry.lookup(name, return_none=True):

            def proc(context: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
                """Processes data using the specified filter.

                Parameters
                ----------
                context : Any
                    The context for the filter.
                data : Any
                    The data to be processed.
                *args : Any
                    Additional arguments for the filter.
                **kwargs : Any
                    Additional keyword arguments for the filter.

                Returns
                -------
                Any
                    The processed data.
                """
                filter = filter_registry.create(name, *args, **kwargs)
                filter.context = context
                return filter.forward(data)

            return proc

    if kind == "sources":
        if source_registry.lookup(name, return_none=True):

            def proc(context: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
                """Processes data using the specified source.

                Parameters
                ----------
                context : Any
                    The context for the source.
                data : Any
                    The data to be processed.
                *args : Any
                    Additional arguments for the source.
                **kwargs : Any
                    Additional keyword arguments for the source.

                Returns
                -------
                Any
                    The processed data.
                """
                source = source_registry.create(name, *args, **kwargs)
                return source.forward(data)

            return proc

    raise ValueError(f"Unknown {kind} '{name}'")
