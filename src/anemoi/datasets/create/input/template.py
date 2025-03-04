# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re
from abc import ABC
from abc import abstractmethod
from functools import wraps
from typing import Any
from typing import Callable
from typing import List

from .context import Context

LOG = logging.getLogger(__name__)


def notify_result(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to notify the context of the result of the method call.

    Parameters
    ----------
    method : Callable[..., Any]
        The method to wrap.

    Returns
    -------
    Callable[..., Any]
        The wrapped method.
    """

    @wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result: Any = method(self, *args, **kwargs)
        self.context.notify_result(self.action_path, result)
        return result

    return wrapper


class Substitution(ABC):
    """Abstract base class for substitutions in templates."""

    @abstractmethod
    def resolve(self, context: Context) -> Any:
        """Resolve the substitution using the given context.

        Parameters
        ----------
        context : Context
            The context to use for resolution.

        Returns
        -------
        Any
            The resolved value.
        """
        pass


class Reference(Substitution):
    """A class to represent a reference to another value in the context."""

    def __init__(self, context: Any, action_path: List[str]) -> None:
        """Initialize a Reference instance.

        Parameters
        ----------
        context : Any
            The context in which the reference exists.
        action_path : list of str
            The action path to resolve.
        """
        self.context: Any = context
        self.action_path: List[str] = action_path

    def resolve(self, context: Context) -> Any:
        """Resolve the reference using the given context.

        Parameters
        ----------
        context : Context
            The context to use for resolution.

        Returns
        -------
        Any
            The resolved value.
        """
        return context.get_result(self.action_path)


def resolve(context: Context, x: Any) -> Any:
    """Recursively resolve substitutions in the given structure using the context.

    Parameters
    ----------
    context : Context
        The context to use for resolution.
    x : Union[tuple, list, dict, Substitution, Any]
        The structure to resolve.

    Returns
    -------
    Any
        The resolved structure.
    """
    if isinstance(x, tuple):
        return tuple([resolve(context, y) for y in x])

    if isinstance(x, list):
        return [resolve(context, y) for y in x]

    if isinstance(x, dict):
        return {k: resolve(context, v) for k, v in x.items()}

    if isinstance(x, Substitution):
        return x.resolve(context)

    return x


def substitute(context: Context, x: Any) -> Any:
    """Recursively substitute references in the given structure using the context.

    Parameters
    ----------
    context : Context
        The context to use for substitution.
    x : Union[tuple, list, dict, str, Any]
        The structure to substitute.

    Returns
    -------
    Any
        The substituted structure.
    """
    if isinstance(x, tuple):
        return tuple([substitute(context, y) for y in x])

    if isinstance(x, list):
        return [substitute(context, y) for y in x]

    if isinstance(x, dict):
        return {k: substitute(context, v) for k, v in x.items()}

    if not isinstance(x, str):
        return x

    if re.match(r"^\${[\.\w\-]+}$", x):
        path = x[2:-1].split(".")
        context.will_need_reference(path)
        return Reference(context, path)

    return x
