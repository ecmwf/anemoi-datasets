# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Callable

from ..source import Source
from . import source_registry


class LegacySource(Source):
    """A legacy source class.

    Parameters
    ----------
    context : Any
        The context in which the source is created.
    *args : tuple
        Positional arguments.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(self, context: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(context, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs


class legacy_source:
    """A decorator class for legacy sources.

    Parameters
    ----------
    name : str
        The name of the legacy source.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, execute: Callable) -> Callable:
        """Call method to wrap the execute function.

        Parameters
        ----------
        execute : function
            The execute function to be wrapped.

        Returns
        -------
        function
            The wrapped execute function.
        """
        name = f"Legacy{self.name.title()}Source"

        def execute_wrapper(self, *args: Any, **kwargs: Any) -> Any:
            """Wrapper method to call the execute function.

            Parameters
            ----------
            *args : tuple
            Positional arguments to pass to the execute function.
            **kwargs : dict
            Keyword arguments to pass to the execute function.
            """
            return execute(self.context, *args, **kwargs)

        klass = type(name, (LegacySource,), {})
        klass.execute = execute_wrapper

        source_registry.register(self.name)(klass)

        return execute
