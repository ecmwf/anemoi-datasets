#!/usr/bin/env python3
# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import glob
import hashlib
import json
import os
import shutil
import warnings
from functools import wraps
from unittest.mock import patch

import numpy as np
import pytest
import requests
from earthkit.data import from_source

from anemoi.datasets import open_dataset
from anemoi.datasets.create import Creator
from anemoi.datasets.data.stores import open_zarr

TEST_DATA_ROOT = "https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/anemoi-datasets/create/"


HERE = os.path.dirname(__file__)
# find_yamls
NAMES = sorted([os.path.basename(path).split(".")[0] for path in glob.glob(os.path.join(HERE, "*.yaml"))])
SKIP = ["recentre"]
NAMES = [name for name in NAMES if name not in SKIP]
assert NAMES, "No yaml files found in " + HERE


def mockup_from_source(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("earthkit.data.from_source", _from_source):
            return func(*args, **kwargs)

    return wrapper


class LoadSource:
    def __init__(self, read_dir=None, write_dir=None):
        self.read_dir = read_dir
        self.write_dir = write_dir

    def filename(self, args, kwargs):
        try:
            string = json.dumps([args, kwargs])
        except Exception as e:
            warnings.warn(f"Could not build hash for {args}, {kwargs}, {e}")
            return None
        h = hashlib.md5(string.encode("utf8")).hexdigest()
        return h + ".copy"

    def write(self, directory, ds, args, kwargs):
        if self.write_dir is None:
            return

        if not hasattr(ds, "path"):
            return
        filename = self.filename(args, kwargs)
        path = os.path.join(directory, filename)
        print(f"Saving to {path} for {args}, {kwargs}")
        shutil.copy(ds.path, path)

    def read(self, directory, args, kwargs):
        if self.read_dir is None:
            return None
        filename = self.filename(args, kwargs)
        if filename is None:
            return None
        path = os.path.join(directory, filename)

        if os.path.exists(path):
            print(f"Mockup: Loading path {path} for {args}, {kwargs}")
            ds = from_source("file", path)
            return ds

        elif path.startswith("http:") or path.startswith("https:"):
            print(f"Mockup: Loading url {path} for {args}, {kwargs}")
            try:
                return from_source("url", path)
            except requests.exceptions.HTTPError:
                print(f"Mockup: ❌ Cannot load from url for {path} for {args}, {kwargs}")

        return None

    def source_name(self, *args, **kwargs):
        if args:
            return args[0]
        return kwargs["name"]

    def __call__(self, *args, **kwargs):
        name = self.source_name(*args, **kwargs)

        if name != "mars":
            return from_source(*args, **kwargs)

        ds = self.read(self.read_dir, args, kwargs)
        if ds is not None:
            return ds

        ds = from_source(*args, **kwargs)

        self.write(self.write_dir, ds, args, kwargs)

        return ds


_from_source = LoadSource(
    read_dir=os.environ.get("LOAD_SOURCE_MOCKUP_READ_DIRECTORY", TEST_DATA_ROOT),
    write_dir=os.environ.get("LOAD_SOURCE_MOCKUP_WRITE_DIRECTORY"),
)


def compare_dot_zattrs(a, b):
    if isinstance(a, dict):
        a_keys = list(a.keys())
        b_keys = list(b.keys())
        for k in set(a_keys) & set(b_keys):
            if k in ["timestamp", "uuid", "latest_write_timestamp", "yaml_config"]:
                assert type(a[k]) == type(b[k]), (  # noqa: E721
                    type(a[k]),
                    type(b[k]),
                    a[k],
                    b[k],
                )
            assert k in a_keys, (k, a_keys)
            assert k in b_keys, (k, b_keys)
            return compare_dot_zattrs(a[k], b[k])

    if isinstance(a, list):
        assert len(a) == len(b), (a, b)
        for v, w in zip(a, b):
            return compare_dot_zattrs(v, w)

    assert type(a) == type(b), (type(a), type(b), a, b)  # noqa: E721
    return a == b, (a, b)


def compare_datasets(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a.dates == b.dates).all(), (a.dates, b.dates)
    for a_, b_ in zip(a.variables, b.variables):
        assert a_ == b_, (a, b)
    assert a.missing == b.missing, "Missing are different"

    for i_date, date in zip(range(a.shape[0]), a.dates):
        if i_date in a.missing:
            continue
        for i_param in range(a.shape[1]):
            param = a.variables[i_param]
            assert param == b.variables[i_param], (
                date,
                param,
                a.variables[i_param],
                b.variables[i_param],
            )
            a_ = a[i_date, i_param]
            b_ = b[i_date, i_param]
            assert a.shape == b.shape, (date, param, a.shape, b.shape)

            a_nans = np.isnan(a_)
            b_nans = np.isnan(b_)
            assert np.all(a_nans == b_nans), (date, param, "nans are different")

            a_ = np.where(a_nans, 0, a_)
            b_ = np.where(b_nans, 0, b_)

            delta = a_ - b_
            max_delta = np.max(np.abs(delta))
            assert max_delta == 0.0, (date, param, a_, b_, a_ - b_, max_delta)


def compare_statistics(ds1, ds2):
    vars1 = ds1.variables
    vars2 = ds2.variables
    assert len(vars1) == len(vars2)
    for v1, v2 in zip(vars1, vars2):
        idx1 = ds1.name_to_index[v1]
        idx2 = ds2.name_to_index[v2]
        assert (ds1.statistics["mean"][idx1] == ds2.statistics["mean"][idx2]).all()
        assert (ds1.statistics["stdev"][idx1] == ds2.statistics["stdev"][idx2]).all()
        assert (ds1.statistics["maximum"][idx1] == ds2.statistics["maximum"][idx2]).all()
        assert (ds1.statistics["minimum"][idx1] == ds2.statistics["minimum"][idx2]).all()


class Comparer:
    def __init__(self, name, output_path=None, reference_path=None):
        self.name = name
        self.reference = reference_path or os.path.join(TEST_DATA_ROOT, name + ".zarr")
        self.output = output_path or os.path.join(name + ".zarr")
        print(f"Comparing {self.reference} and {self.output}")

        self.z_reference = open_zarr(self.reference)
        self.z_output = open_zarr(self.output)

        self.ds_reference = open_dataset(self.reference)
        self.ds_output = open_dataset(self.output)

    def compare(self):
        compare_dot_zattrs(self.z_output.attrs, self.z_reference.attrs)
        compare_datasets(self.ds_output, self.ds_reference)
        compare_statistics(self.ds_output, self.ds_reference)


@pytest.mark.parametrize("name", NAMES)
@mockup_from_source
def test_run(name):
    config = os.path.join(HERE, name + ".yaml")
    output = os.path.join(HERE, name + ".zarr")

    # cache=None is using the default cache
    c = Creator(output, config=config, cache=None, overwrite=True)
    c.init()
    c.load()
    c.finalise()
    c.additions(delta=[1, 3, 6, 12])
    c.cleanup()

    comparer = Comparer(name, output_path=output)
    comparer.compare()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the test case")
    args = parser.parse_args()
    test_run(args.name)
