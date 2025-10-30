# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod
from typing import Any

from anemoi.datasets.create.input.context import Context

from ..source import Source

LOG = logging.getLogger(__name__)


class LegacySource(Source):
    """A legacy source class.

    Parameters
    ----------
    context : Context
        The context in which the source is created.
    *args : tuple
        Positional arguments.
    **kwargs : dict
        Keyword arguments.
    """

    def __init__(self, context: Context, *args: Any, **kwargs: Any) -> None:
        super().__init__(context, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    @abstractmethod
    def _execute(context, *args, **kwargs):
        pass

    def execute(self, dates: Any) -> Any:
        return self._execute(self.context, dates, *self.args, **self.kwargs)
