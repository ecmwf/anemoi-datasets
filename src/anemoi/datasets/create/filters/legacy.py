# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import os
from typing import Any
from typing import Callable

from ..filter import Filter
from . import filter_registry


class LegacyFilter(Filter):
    """A legacy filter class.

    Parameters
    ----------
    context : Any
        The context in which the filter is created.
    *args : tuple
        Positional arguments.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(self, context: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(context, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs


class legacy_filter:
    """A decorator class for legacy filters.

    Parameters
    ----------
    name : str
        The name of the legacy filter.
    """

    def __init__(self, name: str) -> None:
        name, _ = os.path.splitext(os.path.basename(name))
        self.name = name

    def __call__(self, execute: Callable) -> Callable:
        """Call method to wrap the execute function.

        Parameters
        ----------
        execute : Callable
            The execute function to be wrapped.

        Returns
        -------
        Callable
            The wrapped execute function.
        """
        name = f"Legacy{self.name.title()}Filter"

        def execute_wrapper(self, input) -> Any:
            """Wrapper method to call the execute function."""
            return execute(self.context, input, *self.args, **self.kwargs)

        klass = type(name, (LegacyFilter,), {})
        klass.execute = execute_wrapper

        filter_registry.register(self.name)(klass)

        return execute
