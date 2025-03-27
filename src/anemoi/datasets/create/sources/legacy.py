# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import inspect
import logging
import os
from typing import Any
from typing import Callable

from anemoi.datasets.create.input.template import resolve

from ..source import Source
from . import source_registry

LOG = logging.getLogger(__name__)


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
        name, _ = os.path.splitext(os.path.basename(name))
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
        this = self
        name = f"Legacy{self.name.title()}Source"
        source = ".".join([execute.__module__, execute.__name__])

        def execute_wrapper(self, dates) -> Any:
            """Wrapper method to call the execute function."""

            args, kwargs = resolve(self.context, (self.args, self.kwargs))

            try:
                return execute(self.context, dates, *args, **kwargs)
            except TypeError:
                LOG.error(f"Error executing source {this.name} from {source}")
                LOG.error(f"Function signature is: {inspect.signature(execute)}")
                LOG.error(f"Arguments are: {args=}, {kwargs=}")
                raise

        klass = type(
            name,
            (LegacySource,),
            {
                "execute": execute_wrapper,
                "_source": source,
            },
        )

        source_registry.register(self.name)(klass)

        return execute
