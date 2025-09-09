# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import pandas as pd
from bufr2df_parallel import bufr2df_parallel
from earthkit.data import from_source

from anemoi.datasets.create.sources.observations import ObservationsFilter
from anemoi.datasets.create.sources.observations import ObservationsSource
from anemoi.datasets.data.records import AbsoluteWindow
from anemoi.datasets.data.records import window_from_str

log = logging.getLogger(__name__)


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


class MarsObsSource(ObservationsSource):
    def __init__(self, request_dict, pre_process_dict, process_func):
        assert isinstance(request_dict, dict), "request_dict must be a dictionary"
        self.request_dict = request_dict
        self.pre_process_dict = pre_process_dict
        self.process_func = process_func

    def __call__(self, window):
        assert isinstance(window, AbsoluteWindow), "window must be an AbsoluteWindow"

        request_dict = self.request_dict
        request_dict["date"] = f"{window.start.strftime('%Y%m%d')}/to/{window.end.strftime('%Y%m%d')}"
        try:
            ekd_ds = from_source("mars", request_dict)
        except Exception as e:
            if "File is empty" in str(e):
                log.warning(
                    f"Empty file for period {window.start.strftime('%Y%m%d')} to {window.end.strftime('%Y%m%d')}. Skipping."
                )
                return
            else:
                raise  # Re-raise if it's a different error

        data = self.process_func(ekd_ds, **self.pre_process_dict)

        if window.include_start:
            mask = data["times"] > window.start
        else:
            mask = data["times"] >= window.start
        if window.include_end:
            mask &= data["times"] <= window.end
        else:
            mask &= data["times"] < window.end

        df = data[mask]

        return self._check(df)


class ColFilter(ObservationsFilter):
    def __init__(self, col_name):
        self.col_name = col_name

    def __call__(self, df):
        """Filter the data based on the given window."""
        self._check(df)
        # Here we can add any filtering logic if needed
        df.loc[:, self.col_name] = df[self.col_name] + 0.42
        return self._check(df)


dates = [datetime.datetime(2015, 10, 1, 0, 0) + datetime.timedelta(hours=i * 8) for i in range(3)]

source = MarsObsSource(
    request_dict={
        "class": "od",
        "expver": "0001",
        "stream": "DCDA/LWDA",
        "type": "ai",
        "obstype": "ssmis",
        "times": "00/06/12/18",
    },
    pre_process_dict={
        # "target": odb2df.process_odb,
        "nproc": 12,
        "prefilter_msg_header": {"satelliteID": 286.0},
        "datetime_position_prefix": "#1#",
        "per_report": {
            "satelliteID": "satelliteID",
            "#1#latitude": "latitudes",
            "#1#longitude": "longitudes",
            # bearingOrAzimuth: azimuth
            "fieldOfViewNumber": "fov_num",
            "#9#brightnessTemperature": "obsvalue_rawbt_9",
            "#10#brightnessTemperature": "obsvalue_rawbt_10",
            "#11#brightnessTemperature": "obsvalue_rawbt_11",
            "#12#brightnessTemperature": "obsvalue_rawbt_12",
            "#13#brightnessTemperature": "obsvalue_rawbt_13",
            "#14#brightnessTemperature": "obsvalue_rawbt_14",
            "#15#brightnessTemperature": "obsvalue_rawbt_15",
            "#16#brightnessTemperature": "obsvalue_rawbt_16",
            "#17#brightnessTemperature": "obsvalue_rawbt_17",
            "#18#brightnessTemperature": "obsvalue_rawbt_18",
        },
        "filters": {
            "longitudes": "lambda x: np.isfinite(x)",
            "latitudes": "lambda x: np.isfinite(x)",
        },
    },
    process_func=bufr2df_parallel,
)
filter = ColFilter("obsvalue_rawbt_9")

for d in dates:
    window = window_from_str("(-5h, 1h]").to_absolute_window(d)
    print(window.start.strftime("%Y-%m-%d"), window.end.strftime("%Y-%m-%d"))
    d = source(window)
    d = filter(d)
    print(window)
    print(d)
    print(d["satelliteID"].unique())
