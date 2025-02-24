# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from typing import List

from earthkit.data import FieldList

from .misc import assert_fieldlist
from .result import Result
from .trace import trace_datasource

LOG = logging.getLogger(__name__)


class EmptyResult(Result):
    """Class to represent an empty result in the dataset creation process."""

    empty = True

    def __init__(self, context: object, action_path: list, dates: object) -> None:
        """Initializes an EmptyResult instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : list
            The action path.
        dates : object
            The dates object.
        """
        super().__init__(context, action_path + ["empty"], dates)

    @cached_property
    @assert_fieldlist
    @trace_datasource
    def datasource(self) -> FieldList:
        """Returns an empty datasource."""
        from earthkit.data import from_source

        return from_source("empty")

    @property
    def variables(self) -> List[str]:
        """Returns an empty list of variables."""
        return []
