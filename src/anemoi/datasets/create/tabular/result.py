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

        self.frame = frame
        start_date, end_date = argument.start_date, argument.end_date

        # Filter the DataFrame rows between start_date and end_date (inclusive)
        mask = (self.frame["date"] >= start_date) & (self.frame["date"] <= end_date)
        self.frame = self.frame.loc[mask].reset_index(drop=True)

        # Round date to the nearest second
        # Convert "date" to integer seconds since the Unix epoch
        self.frame["date"] = (self.frame["date"].astype("int64") // 10**9).astype(int)

        # Rename columns to use private attribute naming
        self.frame = self.frame.rename(
            columns={
                "date": "__date",
                "latitude": "__latitude",
                "longitude": "__longitude",
            }
        )

        # Create __date and __time columns from the timestamp
        self.frame["__time"] = self.frame["__date"] % 86400
        self.frame["__date"] = self.frame["__date"] // 86400

        # Move __date, __time, __latitude and __longitude to the beginning of the DataFrame
        cols = self.frame.columns.tolist()
        first_cols = ["__date", "__time", "__latitude", "__longitude"]
        other_cols = [col for col in cols if col not in first_cols]
        self.frame = self.frame[first_cols + other_cols]

        # Sort the DataFrame lexicographically by all columns
        self.frame = self.frame.sort_values(by=self.frame.columns.tolist(), kind="mergesort").reset_index(drop=True)
        self.argument = argument

    def to_numpy(self, dtype: type = np.float32) -> np.ndarray:
        # Convert the DataFrame to a 2D NumPy array of type float32
        return self.frame.to_numpy(dtype=dtype)

    @property
    def start_date(self) -> datetime.datetime:
        return self.argument.start_date

    @property
    def end_date(self) -> datetime.datetime:
        return self.argument.end_date

    @property
    def variables(self) -> list[str]:
        return self.frame.columns.tolist()
