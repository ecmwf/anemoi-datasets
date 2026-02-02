# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from anemoi.datasets.create.input.result import Result

LOG = logging.getLogger(__name__)


class TabularResult(Result):
    """Class to represent the result of an action in the dataset creation process."""

    def __init__(self, context: Any, argument: Any, frame: pd.DataFrame) -> None:

        assert isinstance(frame, pd.DataFrame), type(frame)

        assert "latitude" in frame.columns, frame.columns
        assert "longitude" in frame.columns, frame.columns
        assert "date" in frame.columns, frame.columns

        assert np.issubdtype(frame["latitude"].dtype, np.floating)
        assert np.issubdtype(frame["longitude"].dtype, np.floating)
        assert np.issubdtype(frame["date"].dtype, np.datetime64)

        self.frame = frame.reset_index(drop=True)
        start_range, end_range = argument.start_range, argument.end_range

        # Filter the DataFrame rows between start_range and end_range (inclusive)
        mask = (self.frame["date"] >= start_range) & (self.frame["date"] <= end_range)
        start = time.time()
        self.frame = self.frame.loc[mask].reset_index(drop=True)
        LOG.info(
            f"Filtered TabularResult between {start_range} and {end_range} in {time.time() - start:.2f} seconds ({len(self.frame):,} rows)"
        )

        # Round date to the nearest second
        # Convert "date" to integer seconds since the Unix epoch
        start = time.time()
        self.frame["date"] = (self.frame["date"].astype("int64") / 10**9).round().astype("int64")
        LOG.info(
            f"Converted 'date' to integer seconds since epoch in {time.time() - start:.2f} seconds ({len(self.frame):,} rows)"
        )

        # Rename columns to use private attribute naming
        self.frame = self.frame.rename(
            columns={
                "date": "__date",
                "latitude": "__latitude",
                "longitude": "__longitude",
            }
        )

        # Create __date and __time columns from the timestamp
        start = time.time()
        self.frame["__time"] = self.frame["__date"] % 86400
        self.frame["__date"] = self.frame["__date"] // 86400
        LOG.info(f"Created __date and __time columns in {time.time() - start:.2f} seconds ({len(self.frame):,} rows)")

        # Normalize longitudes to be within [0, 360)
        start = time.time()
        self.frame["__longitude"] = self.frame["__longitude"] % 360
        LOG.info(f"Normalised longitudes in {time.time() - start:.2f} seconds ({len(self.frame):,} rows)")

        # Move __date, __time, __latitude and __longitude to the beginning of the DataFrame
        cols = self.frame.columns.tolist()
        first_cols = ["__date", "__time", "__latitude", "__longitude"]
        other_cols = [col for col in cols if col not in first_cols]
        self.frame = self.frame[first_cols + other_cols]

        # Sort the DataFrame lexicographically by all columns
        start = time.time()
        self.frame = self.frame.sort_values(by=self.frame.columns.tolist(), kind="mergesort").reset_index(drop=True)
        LOG.info(f"Sorted TabularResult in {time.time() - start:.2f} seconds ({len(self.frame):,} rows)")

        # Remove duplicate rows
        start = time.time()
        size = len(self.frame)
        self.frame = self.frame.drop_duplicates().reset_index(drop=True)
        dedup_size = len(self.frame)
        LOG.info(f"Deduplicated TabularResult in {time.time() - start:.2f} seconds {size:,} rows")

        if dedup_size < size:
            LOG.warning(f"Removed {size - dedup_size:,} duplicate rows during TabularResult creation")

        self.argument = argument

    def to_numpy(self, dtype: type = np.float32) -> np.ndarray:
        # Convert the DataFrame to a 2D NumPy array of type float32
        start = time.time()
        result = self.frame.to_numpy(dtype=dtype)
        LOG.info(
            f"Converted TabularResult to NumPy array in {time.time() - start:.2f} seconds ({result.shape[0]:,} rows, {result.shape[1]:,} columns)"
        )
        return result

    @property
    def start_range(self) -> datetime.datetime:
        return self.argument.start_range

    @property
    def end_range(self) -> datetime.datetime:
        return self.argument.end_range

    @property
    def variables(self) -> list[str]:
        return self.frame.columns.tolist()
