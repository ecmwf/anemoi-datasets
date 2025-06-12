# (C) Copyright 2025  Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Dict

import earthkit.data as ekd

from ..filter import Filter


class TransformFilter(Filter):
    """Calls filters from anemoi.transform.filters

    Parameters
    ----------
    context : Any
        The context in which the filter is created.
    name : str
        The name of the filter.
    config : Dict[str, Any]
        The configuration for the filter.
    """

    def __init__(self, context: Any, name: str, config: Dict[str, Any]) -> None:

        from anemoi.transform.filters import create_filter

        self.name = name
        self.transform_filter = create_filter(context, config)

    def execute(self, input: ekd.FieldList) -> ekd.FieldList:
        """Execute the transformation filter.

        Parameters
        ----------
        input : ekd.FieldList
            The input data to be transformed.

        Returns
        -------
        ekd.FieldList
            The transformed data.
        """
        return self.transform_filter.forward(input)
