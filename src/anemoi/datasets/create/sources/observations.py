# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pandas as pd


def check_dataframe(df):
    """Check the DataFrame for consistency."""
    if df.empty:
        pass
    if "times" not in df.columns:
        raise ValueError("The DataFrame must contain a 'times' column.")
    if not pd.api.types.is_datetime64_any_dtype(df["times"]):
        raise TypeError("The 'times' column must be of datetime type.")
    if "latitudes" not in df.columns or "longitudes" not in df.columns:
        raise ValueError("The DataFrame must contain 'latitudes' and 'longitudes' columns.")


class ObservationsSource:
    def __call__(self, window):
        raise NotImplementedError("This method should be implemented by subclasses")

    def _check(self, df):
        check_dataframe(df)
        return df


class ObservationsFilter:
    def __call__(self, df):
        """Filter the data based on the given window."""
        check_dataframe(df)
        return df

    def _check(self, df):
        check_dataframe(df)
        return df
