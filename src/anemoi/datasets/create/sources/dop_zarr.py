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

        self.lat_idx = self.data.attrs["colnames"].index("lat")
        self.lon_idx = self.data.attrs["colnames"].index("lon")

        self.colnames = ["date", "latitude", "longitude"] + [
            col for col in self.data.attrs["colnames"] if col not in ["lat", "lon"]
        ]
        self.dtypes = {col: "float32" for col in self.colnames}
        self.dtypes["date"] = "datetime64[ns]"

    def execute(self, dates):
        import pandas as pd

        LOG.info(f"Search start index for date {type(dates)}")

        # Cannot use np.searchsorted because dates is 2D

        """A proxy to access only the first dimension of the 2D dates array."""

        class Proxy:
            def __init__(self, dates):
                self.dates = dates

            def __len__(self):
                return len(self.dates)

            def __getitem__(self, value):
                return self.dates[value][0]

        # Search only the first dimension of the 2D dates array, without loading all dates
        start_idx = bisect.bisect_left(Proxy(self.dates), np.datetime64(dates.start_date))
        end_idx = bisect.bisect_right(Proxy(self.dates), np.datetime64(dates.end_date))

        if start_idx >= end_idx:
            LOG.warning(f"No data found between {dates.start_date} and {dates.end_date}")
            return pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in self.dtypes.items()})

        # Load data slice
        LOG.info(
            f"Loading {dates.start_date} => {dates.end_date} slice[{start_idx}:{end_idx}] -> {end_idx - start_idx} records"
        )
        data_slice = self.data[start_idx:end_idx]

        # Build data frame

        date_slice = self.dates[start_idx:end_idx]
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

        return frame
