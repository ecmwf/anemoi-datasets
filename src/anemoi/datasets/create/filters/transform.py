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

    def __init__(self, context: Any, name: str, config: Dict[str, Any]) -> None:
        super().__init__(context)
        from anemoi.transform.filters import create_filter

        self.name = name
        self.transform_filter = create_filter(config)

    def execute(self, context: Any, input: ekd.FieldList) -> ekd.FieldList:
        return self.transform_filter.forward(input)
