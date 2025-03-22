# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import wraps
from typing import Any
from typing import Callable
from typing import Tuple
from typing import Union

from earthkit.data import FieldList
from earthkit.data.core.fieldlist import MultiFieldList

LOG = logging.getLogger(__name__)


def parse_function_name(name: str) -> Tuple[str, Union[int, None]]:
    """Parses a function name to extract the base name and an optional time delta.

    Parameters
    ----------
    name : str
        The function name to parse.

    Returns
    -------
    tuple of (str, int or None)
        The base name and an optional time delta.
    """
    if name.endswith("h") and name[:-1].isdigit():

        if "-" in name:
            name, delta = name.split("-")
            sign = -1

        elif "+" in name:
            name, delta = name.split("+")
            sign = 1

        else:
            return name, None

        assert delta[-1] == "h", (name, delta)
        delta = sign * int(delta[:-1])
        return name, delta

    return name, None


def assert_fieldlist(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to assert that the result of a method is an instance of FieldList.

    Parameters
    ----------
    method : Callable[..., Any]
        The method to decorate.

    Returns
    -------
    Callable[..., Any]
        The decorated method.
    """

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:

        result = method(self, *args, **kwargs)
        assert isinstance(result, FieldList), type(result)
        return result

    return wrapper


def assert_is_fieldlist(obj: object) -> None:
    """Asserts that the given object is an instance of FieldList.

    Parameters
    ----------
    obj : object
        The object to check.
    """
    assert isinstance(obj, FieldList), type(obj)


def _flatten(ds: Union[MultiFieldList, FieldList]) -> list:
    """Flattens a MultiFieldList or FieldList into a list of FieldList objects.

    Parameters
    ----------
    ds : Union[MultiFieldList, FieldList]
        The dataset to flatten.

    Returns
    -------
    list
        A list of FieldList objects.
    """
    if isinstance(ds, MultiFieldList):
        return [_tidy(f) for s in ds._indexes for f in _flatten(s)]
    return [ds]


def _tidy(ds: Union[MultiFieldList, FieldList], indent: int = 0) -> Union[MultiFieldList, FieldList]:
    """Tidies up a MultiFieldList or FieldList by removing empty sources.

    Parameters
    ----------
    ds : Union[MultiFieldList, FieldList]
        The dataset to tidy.
    indent : int, optional
        The indentation level. Defaults to 0.

    Returns
    -------
    Union[MultiFieldList, FieldList]
        The tidied dataset.
    """
    if isinstance(ds, MultiFieldList):

        sources = [s for s in _flatten(ds) if len(s) > 0]
        if len(sources) == 1:
            return sources[0]
        return MultiFieldList(sources)
    return ds
