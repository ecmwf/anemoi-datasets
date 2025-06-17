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
from earthkit.data import from_source
from odb2df import process_odb

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


class MarsSource(ObservationsSource):
    def __init__(self, request_dict, post_process_dict):
        assert isinstance(request_dict, dict), "request_dict must be a dictionary"
        self.request_dict = request_dict
        self.post_process_dict = post_process_dict

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

        data = process_odb(ekd_ds, **self.post_process_dict)

        print(data)
        print(data.columns)

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


class DummyFilter(ObservationsFilter):
    def __call__(self, df, col_name):
        """Filter the data based on the given window."""
        self._check(df)
        # Here we can add any filtering logic if needed
        df.loc[:, col_name] = df[col_name] + 0.42
        return self._check(df)


dates = [datetime.datetime(2025, 1, 1, 0, 0) + datetime.timedelta(hours=i * 8) for i in range(3)]

N = 100
source = MarsSource(
    request_dict={
        "class": "ea",
        "expver": "0001",
        "stream": "oper",
        "obsgroup": "conv",
        "reportype": "16001/16002/16004/16065/16076",
        "type": "ofb",
        "time": "00/12",
        "filter": "'select seqno,reportype,date,time,lat,lon,report_status,report_event1,entryno,varno,statid,stalt,obsvalue,lsm@modsurf,biascorr_fg,final_obs_error,datum_status@body,datum_event1@body,vertco_reference_1,vertco_type where ((varno==39 and abs(fg_depar@body)<20) or (varno in (41,42) and abs(fg_depar@body)<15) or (varno==58 and abs(fg_depar@body)<0.4) or (varno == 110 and entryno == 1 and abs(fg_depar@body)<10000) or (varno == 91)) and time in (000000,030000,060000,090000,120000,150000,180000,210000);'",
    },
    post_process_dict={
        "index": ["seqno@hdr", "lat@hdr", "lon@hdr", "date@hdr", "time@hdr", "stalt@hdr", "lsm@modsurf"],
        "pivot": ["varno@body"],
        "values": ["obsvalue@body"],
    },
)
filter = DummyFilter()

for d in dates:
    window = window_from_str("(-5h, 1h]").to_absolute_window(d)
    print(window.start.strftime("%Y-%m-%d"), window.end.strftime("%Y-%m-%d"))
    d = source(window)
    d = filter(d, "obsvalue_v10m_0")
    print(window)
    print(d)
