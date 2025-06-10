# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import pandas as pd

from anemoi.datasets.create.sources.observations import ObservationsFilter
from anemoi.datasets.create.sources.observations import ObservationsSource
from anemoi.datasets.data.records import AbsoluteWindow
from anemoi.datasets.data.records import window_from_str


class DummpySource(ObservationsSource):
    def __init__(self, data):
        assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
        self.data = data

    def __call__(self, window):
        assert isinstance(window, AbsoluteWindow), "window must be an AbsoluteWindow"

        if window.include_start:
            mask = self.data["times"] > window.start
        else:
            mask = self.data["times"] >= window.start
        if window.include_end:
            mask &= self.data["times"] <= window.end
        else:
            mask &= self.data["times"] < window.end

        df = self.data[mask]

        return self._check(df)


class DummyFilter(ObservationsFilter):
    def __call__(self, df):
        """Filter the data based on the given window."""
        self._check(df)
        # Here we can add any filtering logic if needed
        df["a1"] = df["a1"] + 0.42
        return self._check(df)


dates = [datetime.datetime(2023, 1, 1, 0, 0) + datetime.timedelta(hours=i * 8) for i in range(3)]

N = 100
source = DummpySource(
    pd.DataFrame(
        {
            "times": np.arange(N) * datetime.timedelta(hours=1) + dates[0],
            "latitudes": -0.1 * np.arange(N),
            "longitudes": -0.2 * np.arange(N),
            "a1": np.arange(N) * 1.0,
            "a2": np.arange(N) * 2.0,
        }
    )
)
filter = DummyFilter()

for d in dates:
    window = window_from_str("(-5h, 1h]").to_absolute_window(d)
    d = source(window)
    d = filter(d)
    print(window)
    print(d)
