# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import bisect
import logging
import time

import numpy as np

from ..source import Source
from . import source_registry

LOG = logging.getLogger(__name__)


@source_registry.register("dop-zarr")
class DOPZarrSource(Source):
    """A source that reads Zarr from the DOP project."""

    emoji = "📄"  # For tracing

    def __init__(
        self,
        context: any,
        path: str,
        columns: list = None,
        *args,
        **kwargs,
    ):

        import zarr

        super().__init__(context, *args, **kwargs)

        self.path = path
        self.columns = columns
        self.store = zarr.open(self.path, mode="r")
        self.dates = self.store["dates"]
        self.data = self.store["data"]

        self.latitude = None
        self.longitude = None

        for col in (("lat", "lon"), ("latitude", "longitude")):
            if all(c in self.data.attrs["colnames"] for c in col):
                if self.latitude is not None or self.longitude is not None:
                    raise ValueError(
                        f"Found multiple latitude/longitude column pairs in data: {self.latitude}/{self.longitude} and {col[0]}/{col[1]}"
                    )
                self.latitude, self.longitude = col

        self.lat_idx = self.data.attrs["colnames"].index(self.latitude)
        self.lon_idx = self.data.attrs["colnames"].index(self.longitude)

        self.colnames = ["date", "latitude", "longitude"] + [
            col for col in self.data.attrs["colnames"] if col not in [self.latitude, self.longitude]
        ]
        self.dtypes = {col: "float32" for col in self.colnames}
        self.dtypes["date"] = "datetime64[ns]"

    def execute(self, dates):
        import pandas as pd

        start = time.time()
        LOG.info(
            f"Loading {dates.start_range} => {dates.end_range} ({(dates.end_range - dates.start_range).astype('timedelta64[s]').astype(object)})"
        )
        # Cannot use np.searchsorted because dates is 2D

        """A proxy to access only the first dimension of the 2D dates array."""

        class Proxy:
            def __init__(self, dates):
                self.dates = dates

            def __len__(self):
                return len(self.dates)

            def __getitem__(self, value):
                return self.dates[value][0]

        start = time.time()
        # Search only the first dimension of the 2D dates array, without loading all dates
        start_idx = bisect.bisect_left(Proxy(self.dates), np.datetime64(dates.start_range))
        end_idx = bisect.bisect_right(Proxy(self.dates), np.datetime64(dates.end_range))
        date_lookup_time = time.time() - start

        if start_idx >= end_idx:
            LOG.warning(f"No data found between {dates.start_range} and {dates.end_range}")
            return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in self.dtypes.items()})

        # Load data slice
        LOG.info(
            f"Loading {dates.start_range} => {dates.end_range} slice[{start_idx}:{end_idx}] -> {end_idx - start_idx:,} records"
        )

        start = time.time()
        data_slice = self.data[start_idx:end_idx]
        data_load_time = time.time() - start

        # Build data frame
        start = time.time()
        date_slice = self.dates[start_idx:end_idx]
        date_load_time = time.time() - start

        LOG.info(
            f"Loaded data slice with {len(data_slice):,} records in {data_load_time:.2f} seconds, date slice in {date_load_time:.2f} seconds"
        )

        LOG.info(f"First date in slice: {date_slice[0][0]}, last date in slice: {date_slice[-1][0]}")
        LOG.info(f"Requested date range: {dates.start_range} to {dates.end_range}")

        assert date_slice[0][0] >= np.datetime64(
            dates.start_range
        ), "First date in slice is before requested start date"
        assert date_slice[-1][0] <= np.datetime64(dates.end_range), "Last date in slice is after requested end date"

        start = time.time()
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(date_slice[:, 0]),
                "latitude": data_slice[:, self.lat_idx].astype(float),
                "longitude": data_slice[:, self.lon_idx].astype(float),
                **{
                    col: data_slice[:, idx].astype(float)
                    for idx, col in enumerate(self.data.attrs["colnames"])
                    if col not in ["lat", "lon"]
                },
            }
        )
        data_frame_time = time.time() - start
        total_time = date_lookup_time + data_load_time + date_load_time + data_frame_time

        LOG.info(
            f"Loaded data frame with {len(frame):,} records in {total_time:.2f} seconds ({date_lookup_time:.2f}s dates lookup, {data_load_time:.2f}s data load, {date_load_time:.2f}s date load, {data_frame_time:.2f}s frame build)"
        )

        return frame
