# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list

from anemoi.datasets.create.input.repeated_dates import DateMapper
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry

LOG = logging.getLogger(__name__)


@source_registry.register("repeated_dates")
class RepeatedDatesSource(Source):

    def __init__(self, context, source: Any, mode: str, **kwargs) -> None:
        # assert False, (context, source, mode, kwargs)
        super().__init__(context, **kwargs)
        self.mapper = DateMapper.from_mode(mode, source, kwargs)
        self.source = source

    def execute(self, group_of_dates):
        source = self.context.create_source(self.source, "data_sources", str(id(self)))

        result = []
        for one_date_group, many_dates_group in self.mapper.transform(group_of_dates):
            print(f"one_date_group: {one_date_group}, many_dates_group: {many_dates_group}")
            source_results = source(self.context, one_date_group)
            for field in source_results:
                for date in many_dates_group:
                    result.append(new_field_with_valid_datetime(field, date))

        return new_fieldlist_from_list(result)
