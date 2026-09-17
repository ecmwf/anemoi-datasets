# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
from anemoi.transform.fields import NewDataField
from anemoi.transform.fields import NewValidDateTimeField
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.utils.dates import as_timedelta

from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry
from anemoi.datasets.dates.groups import GroupOfDates

LOG = logging.getLogger(__name__)


@source_registry.register("average")
class AverageSource(Source):
    """A source computing the time-average of a sub-source over specified time offsets."""

    def __init__(self, context: Any, what: list, source: dict) -> None:
        super().__init__(context)
        self.what = [as_timedelta(w) for w in what]
        self.source = source

    def execute(self, group_of_dates: GroupOfDates) -> Any:
        """Fetch fields at offset dates and return their time-average per output date.

        Parameters
        ----------
        group_of_dates : GroupOfDates
            The output dates for which to compute averages.

        Returns
        -------
        Any
            A FieldList containing one averaged field per variable per output date.
        """
        if not isinstance(group_of_dates, GroupOfDates):
            raise ValueError("Source 'average' works only for standard group of dates")

        # 1. Build deduplicated high-frequency dates
        seen = set()
        unique_dates = []
        for d in group_of_dates:
            for w in self.what:
                t = d + w
                if t not in seen:
                    seen.add(t)
                    unique_dates.append(t)

        high_freq_group = GroupOfDates(unique_dates, group_of_dates.provider)

        # 2. Fetch fields from sub-source
        action = self.context.create_source(self.source, "data_sources", str(id(self)))
        ds = action(self.context, high_freq_group)

        # 3. Index by valid_datetime
        date_to_fields = defaultdict(list)
        for field in ds:
            vdt = field.metadata("valid_datetime")
            if isinstance(vdt, str):
                vdt = datetime.fromisoformat(vdt)
            date_to_fields[vdt].append(field)

        # 4. Average per output date
        result = []
        for d in group_of_dates:
            fields_for_date = []
            for w in self.what:
                fields_for_date.extend(date_to_fields.get(d + w, []))

            groups = defaultdict(list)
            for field in fields_for_date:
                key = field.metadata(namespace="mars")
                key.pop("date", None)
                key.pop("time", None)
                key.pop("step", None)
                key = frozenset(key.items())
                groups[key].append(field)

            for group_fields in groups.values():
                arrays = [f.to_numpy(flatten=True) for f in group_fields]
                avg_data = np.mean(arrays, axis=0)
                new_field = NewValidDateTimeField(NewDataField(group_fields[0], avg_data), d)
                result.append(new_field)

        return new_fieldlist_from_list(result)
