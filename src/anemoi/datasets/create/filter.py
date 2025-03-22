# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod
from typing import Any

import earthkit.data as ekd


class Filter(ABC):
    """A base class for filters."""

    def __init__(self, context: Any, *args: Any, **kwargs: Any) -> None:
        """Initialise the filter.

        Parameters
        ----------
        context : Any
            The context in which the filter is created.
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """

        self.context = context

    @abstractmethod
    def execute(self, data: ekd.FieldList) -> ekd.FieldList:
        """Execute the filter.

        Parameters
        ----------
        data : ekd.FieldList
            The input data.

        Returns
        -------
        ekd.FieldList
            The output data.
        """

        pass
