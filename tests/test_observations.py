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


@pytest.mark.skipif(not os.environ.get("SLOW_TESTS"), reason="No SLOW_TESTS env var")
def test_read():
    cfg = {"dataset": "observations-ea-ofb-0001-2004-2023-combined-metar-v1"}

    print(f"testing {cfg}")
    ds = open_dataset(cfg)
    show(ds, "2017-11")

    assert len(ds.dates) == 28009, len(ds.dates)
    assert len(ds) == len(ds.dates), (len(ds), len(ds.dates))

    assert ds.dates[0] == np.datetime64("2004-03-30T00:00:00"), ds.dates[0]
    assert ds.dates[1] == np.datetime64("2004-03-30T06:00:00"), ds.dates[1]
    assert ds.dates[-1] == np.datetime64("2023-06-01T00:00:00"), ds.dates[-1]

    assert ds.frequency == datetime.timedelta(seconds=21600), ds.frequency
    assert isinstance(ds.variables, (list, tuple)), type(ds.variables)
    assert len(ds.variables) == 20, ds.variables

    assert isinstance(ds.name_to_index, dict), ds.name_to_index
    assert all(k in ds.name_to_index for k in ds.variables), ds.name_to_index

    assert isinstance(ds.statistics, dict), ds.statistics
    assert isinstance(ds.statistics["mean"], np.ndarray), ds.statistics["mean"]
    assert isinstance(ds.statistics["stdev"], np.ndarray), ds.statistics["stdev"]
    assert isinstance(ds.statistics["minimum"], np.ndarray), ds.statistics["minimum"]
    assert isinstance(ds.statistics["maximum"], np.ndarray), ds.statistics["maximum"]


def show(ds, filter):
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
        cfg = {"dataset": "observations-ea-ofb-0001-2004-2023-combined-metar-v1"}
    ds = open_dataset(cfg)
    show(ds, filter)
