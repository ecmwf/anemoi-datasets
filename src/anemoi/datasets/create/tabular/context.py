# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from anemoi.datasets.create.input.context import Context

from .result import TabularResult


class TabularContext(Context):
    """Context for building tabular output data.

    This class extends the base Context to provide logic for tabular datasets.
    """

    def create_result(self, argument: Any, data: Any) -> TabularResult:
        """Create a TabularResult object for the given argument and data.

        Parameters
        ----------
        argument : Any
            The argument used to create the result.
        data : Any
            The data to be wrapped in the result.

        Returns
        -------
        TabularResult
            The created TabularResult instance.
        """
        return TabularResult(self, argument, data)
