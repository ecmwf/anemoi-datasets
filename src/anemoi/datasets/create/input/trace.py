# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import textwrap
import threading
from functools import wraps
from typing import Any
from typing import Callable

LOG = logging.getLogger(__name__)


thread_local = threading.local()
TRACE = 0


def enable_trace(on_off: int) -> None:
    """Enables or disables tracing.

    Parameters
    ----------
    on_off : int
        1 to enable tracing, 0 to disable.
    """
    global TRACE
    TRACE = on_off


def step(action_path: list[str]) -> str:
    """Returns a formatted string representing the action path.

    Parameters
    ----------
    action_path : list of str
        The action path.

    Returns
    -------
    str
        The formatted action path.
    """
    return f"[{'.'.join(action_path)}]"


def trace(emoji: str, *args: Any) -> None:
    """Logs a trace message with an emoji.

    Parameters
    ----------
    emoji : str
        The emoji to use.
    *args : Any
        The arguments to log.
    """
    if not TRACE:
        return

    if not hasattr(thread_local, "TRACE_INDENT"):
        thread_local.TRACE_INDENT = 0

    print(emoji, " " * thread_local.TRACE_INDENT, *args)


def trace_datasource(method: Callable) -> Callable:
    """Decorator to trace the datasource method.

    Parameters
    ----------
    method : Callable
        The method to decorate.

    Returns
    -------
    Callable
        The wrapped method.
    """

    @wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:

        if not hasattr(thread_local, "TRACE_INDENT"):
            thread_local.TRACE_INDENT = 0

        trace(
            "🌍",
            "=>",
            step(self.action_path),
            self._trace_datasource(*args, **kwargs),
        )
        thread_local.TRACE_INDENT += 1
        result = method(self, *args, **kwargs)
        thread_local.TRACE_INDENT -= 1
        trace(
            "🍎",
            "<=",
            step(self.action_path),
            textwrap.shorten(repr(result), 256),
        )
        return result

    return wrapper


def trace_select(method: Callable) -> Callable:
    """Decorator to trace the select method.

    Parameters
    ----------
    method : Callable
        The method to decorate.

    Returns
    -------
    Callable
        The wrapped method.
    """

    @wraps(method)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(thread_local, "TRACE_INDENT"):
            thread_local.TRACE_INDENT = 0
        trace(
            "👓",
            "=>",
            ".".join(self.action_path),
            self._trace_select(*args, **kwargs),
        )
        thread_local.TRACE_INDENT += 1
        result = method(self, *args, **kwargs)
        thread_local.TRACE_INDENT -= 1
        trace(
            "🍍",
            "<=",
            ".".join(self.action_path),
            textwrap.shorten(repr(result), 256),
        )
        return result

    return wrapper
