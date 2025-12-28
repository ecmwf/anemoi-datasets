# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import bisect
import datetime
import logging
from functools import cached_property

import numpy as np
import tqdm

from ..caching import ChunksCache
from . import DateIndexing
from . import date_indexing_registry

LOG = logging.getLogger(__name__)


@date_indexing_registry.register("bisect")
class DateBisect(DateIndexing):
    name = "bisect"

    def __init__(self, /, store, **kwargs):
        self.store = store

    def bulk_load(self, dates_ranges: np.ndarray) -> None:

        row_size = dates_ranges.nbytes // len(dates_ranges)
        chunck_size = 64 * 1024 * 1024 // row_size  # Adjust chunk size to approx 64MB
        LOG.info(f"Bulk loading {dates_ranges.shape} with chunk size {chunck_size}")
        date_index_ranges = self.store.create_dataset(
            "date_index_ranges",
            # data=dates_ranges,
            shape=dates_ranges.shape,
            dtype=dates_ranges.dtype,
            chunks=(chunck_size, 3),
            overwrite=True,
        )
        date_index_ranges.attrs["_ARRAY_DIMENSIONS"] = ["epoch", "start_idx", "length"]

        with ChunksCache(date_index_ranges) as index:
            total = dates_ranges.shape[0]
            step = chunck_size
            for i in tqdm.tqdm(range(0, total, step)):
                last = min(step, total - i)
                index[i : i + last, :] = dates_ranges[i : i + last, :]

    @cached_property
    def index(self):
        return ChunksCache(self.store["date_index_ranges"])

    def start_end_dates(self) -> tuple[datetime.datetime, datetime.datetime]:
        first_key, last_key = self.index[0][0], self.index[-1][0]
        return datetime.datetime.fromtimestamp(first_key), datetime.datetime.fromtimestamp(last_key)

    def boundaries(self, start: int, end: int) -> tuple[int, int]:
        """A proxy to access only the first dimension of the 2D dates array."""

        class Proxy:
            def __init__(self, dates):
                self.dates = dates

            def __len__(self):
                return len(self.dates)

            def __getitem__(self, value):
                return self.dates[value][0]

        # Search only the first dimension of the 2D dates array, without loading all dates
        start_idx = bisect.bisect_left(Proxy(self.index), start)
        end_idx = bisect.bisect_right(Proxy(self.index), end)

        first_row = self.index[start_idx] if start_idx < len(self.index) else None
        last_row = self.index[end_idx - 1] if end_idx > 0 else None

        return (first_row[0], first_row[1:]), (last_row[0], last_row[1:])


# (49, 23)
# (129, 23)
# (127, 23)
# (128, 23)
# (129, 23)
# (128, 23)
