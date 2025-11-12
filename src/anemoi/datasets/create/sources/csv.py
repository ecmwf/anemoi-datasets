# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from anemoi.datasets.create.gridded.typing import DateList

from ..source import Source
from . import source_registry


@source_registry.register("csv")
class CsvSource(Source):
    """CSV data source."""

    emoji = "?"

    def __init__(
        self,
        context,
        **kwargs: dict[str, Any],
    ):

        super().__init__(context)

    def execute(self, dates: DateList):
        raise NotImplementedError("To be developed")
