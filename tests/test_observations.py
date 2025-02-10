#!/usr/bin/env python3

# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import os

import numpy as np
import pytest

from anemoi.datasets import open_dataset


def str_(t):
    """Not needed, but useful for debugging"""
    import numpy as np

    if isinstance(t, (list, tuple)):
        return "[" + " , ".join(str_(e) for e in t) + "]"
    if isinstance(t, np.ndarray):
        return str(t.shape).replace(" ", "").replace(",", "-")
    if isinstance(t, dict):
        return "{" + " , ".join(f"{k}: {str_(v)}" for k, v in t.items()) + "}"
    return str(t)


# @pytest.mark.skipif(not os.environ.get("SLOW_TESTS"), reason="No SLOW_TESTS env var")
# def test_read_metar():
#     cfg = {"dataset": "observations-ea-ofb-0001-2004-2023-combined-metar-v1"}
#
#     print(f"testing {cfg}")
#     ds = open_dataset(cfg)
#     show(ds, "2017-11")
#
#     assert len(ds.dates) == 28009, len(ds.dates)
#     assert len(ds) == len(ds.dates), (len(ds), len(ds.dates))
#
#     assert ds.dates[0] == np.datetime64("2004-03-30T00:00:00"), ds.dates[0]
#     assert ds.dates[1] == np.datetime64("2004-03-30T06:00:00"), ds.dates[1]
#     assert ds.dates[-1] == np.datetime64("2023-06-01T00:00:00"), ds.dates[-1]
#
#     assert ds.frequency == datetime.timedelta(seconds=21600), ds.frequency
#     assert isinstance(ds.variables, (list, tuple)), type(ds.variables)
#     assert len(ds.variables) == 20, ds.variables
#
#     assert isinstance(ds.name_to_index, dict), ds.name_to_index
#     assert all(k in ds.name_to_index for k in ds.variables), ds.name_to_index
#
#     assert isinstance(ds.statistics, dict), ds.statistics
#     assert isinstance(ds.statistics["mean"], np.ndarray), ds.statistics["mean"]
#     assert isinstance(ds.statistics["stdev"], np.ndarray), ds.statistics["stdev"]
#     assert isinstance(ds.statistics["minimum"], np.ndarray), ds.statistics["minimum"]
#     assert isinstance(ds.statistics["maximum"], np.ndarray), ds.statistics["maximum"]


@pytest.mark.skipif(not os.environ.get("SLOW_TESTS"), reason="No SLOW_TESTS env var")
def test_read_and_check():
    cfg = {"dataset": "observations-ea-ofb-0001-2007-2021-metop-a-iasi-radiances-v1"}

    print(f"testing {cfg}")
    ds = open_dataset(cfg)
    show(ds, "2017-11")

    assert len(ds.dates) == 21226, len(ds.dates)
    assert len(ds) == len(ds.dates), (len(ds), len(ds.dates))

    assert ds.dates[0] == np.datetime64("2007-02-28T00:00:00"), ds.dates[0]
    assert ds.dates[1] == np.datetime64("2007-02-28T06:00:00"), ds.dates[1]
    assert ds.dates[-1] == np.datetime64("2021-09-08T06:00:00"), ds.dates[-1]

    assert ds.frequency == datetime.timedelta(seconds=21600), ds.frequency
    assert isinstance(ds.variables, (list, tuple)), type(ds.variables)
    assert len(ds.variables) == 34, ds.variables

    assert isinstance(ds.name_to_index, dict), ds.name_to_index
    assert all(k in ds.name_to_index for k in ds.variables), ds.name_to_index

    assert isinstance(ds.statistics, dict), ds.statistics
    assert isinstance(ds.statistics["mean"], np.ndarray), ds.statistics["mean"]
    assert isinstance(ds.statistics["stdev"], np.ndarray), ds.statistics["stdev"]
    assert isinstance(ds.statistics["minimum"], np.ndarray), ds.statistics["minimum"]
    assert isinstance(ds.statistics["maximum"], np.ndarray), ds.statistics["maximum"]


def show(ds, filter=None):
    print(ds)
    print(ds.tree())
    print(f"✅ Initialized Observations Dataset with {len(ds)} items")
    print(f"Dates: {ds.dates[0]}, {ds.dates[1]}, ..., {ds.dates[-2]}, {ds.dates[-1]}")
    print(f"Frequency: {ds.frequency}")
    print(f"Variables: {ds.variables}")
    print(f"Name to index: {ds.name_to_index}")
    print("Statistics:")
    for k, v in ds.statistics.items():
        print(f"  {k}: {','.join([str(_) for _ in v])}")

    count = 10
    for i in range(len(ds)):
        date = ds.dates[i]
        if filter and not str(date).startswith(filter):
            continue

        data = ds[i]

        count -= 1
        print(f"✅ Got item {i} for time window ending {date}: {str_(data)}")
        if count == 0:
            break

    print("----------------------------")


DATASETS = [
    "observations-od-ai-0001-2013-2023-amsr2-h180-v2",
    "observations-ea-ofb-0001-1998-2023-noaa-15-amsua-radiances-v1",
    "observations-ea-ofb-0001-2005-2023-noaa-18-amsua-radiances-v1",
    "observations-ea-ofb-0001-2006-2021-metop-a-amsua-radiances-v1",
    "observations-ea-ofb-0001-2012-2023-metop-b-amsua-radiances-v1",
    "observations-ea-ofb-0001-2007-2021-metop-a-iasi-radiances-v1",
    "observations-ea-ofb-0001-2013-2023-metop-b-iasi-radiances-v1",
    "observations-ea-ofb-0001-2019-2023-metop-c-iasi-radiances-v1",
    "observations-ea-ofb-0001-2002-2023-aqua-airs-radiances-v1",
    "observations-ea-ofb-0001-2009-2023-dmsp-17-ssmis-radiances-all-sky-v1",
    "observations-ea-ofb-0001-2012-2023-npp-atms-radiances-v2",
    "observations-ea-ofb-0001-2018-2023-noaa-20-atms-radiances-v1",
    "observations-ea-ofb-0001-2012-2023-npp-cris-radiances-v1",
    "observations-ea-ofb-0001-2014-2023-saral-ralt-wave-v1",
    "observations-od-ec-0001-2007-2012-meteosat-9-seviri-v3",
    "observations-od-ai-0001-2012-2018-meteosat-10-seviri-v3",
    "observations-od-ai-0001-2017-2022-meteosat-8-iodc-seviri-v3",
    "observations-od-ec-0001-2004-2007-meteosat-8-seviri-v3",
    "observations-od-ai-0001-2018-2023-meteosat-11-seviri-v1",
    "observations-ea-ofb-0001-2008-2021-metop-a-gpsro-v2-sort",
    "observations-ea-ofb-0001-2012-2023-metop-b-gpsro-v2-sort",
    "observations-ea-ofb-0001-2018-2023-metop-c-gpsro-v2-sort",
    "observations-ea-ofb-0001-2007-2021-metop-a-ascat-v1",
    "observations-ea-ofb-0001-2013-2023-metop-b-ascat-v1",
    "observations-ea-ofb-0001-2020-2023-metop-c-ascat-v1",
    "observations-ea-ofb-0001-1979-2023-combined-surface-v2",
    "observations-od-ofb-0001-2014-2023-combined-snow-depth-v1",
    "observations-ea-ofb-0001-1979-2023-combined-upper-air-v1",
    "observations-od-ai-0001-2013-2024-nexrad-h220-v1",
]


@pytest.mark.skipif(not os.environ.get("SLOW_TESTS"), reason="No SLOW_TESTS env var")
@pytest.mark.parametrize("dataset", DATASETS)
def test_read(dataset):
    cfg = {"dataset": dataset}

    print(f"testing {cfg}")
    ds = open_dataset(cfg)
    show(ds)


if __name__ == "__main__":
    import argparse

    import yaml

    HERE = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Dataset name or path to yaml config file")
    parser.add_argument("--filter", help="filter dates (ex: 2017 or 2017-11)")
    args = parser.parse_args()

    filter = args.filter

    if args.config:
        cfg = args.config
        if cfg.endswith(".yaml"):
            print(f"Using config file: {cfg}")
            with open(cfg, "r") as f:
                cfg = yaml.safe_load(f)
    else:
        cfg = {"dataset": "observations-ea-ofb-0001-1979-2023-combined-surface-v2"}
    ds = open_dataset(cfg)
    show(ds, filter)
