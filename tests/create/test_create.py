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
from functools import wraps
from unittest.mock import patch

import numpy as np
import pytest
import requests
from earthkit.data import from_source as original_from_source
from multiurl import download

from anemoi.datasets import open_dataset
from anemoi.datasets.create import creator_factory
from anemoi.datasets.data.stores import open_zarr

TEST_DATA_ROOT = "https://object-store.os-api.cci1.ecmwf.int/ml-tests/test-data/anemoi-datasets/create"


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

    def filename(self, args, kwargs):
        string = json.dumps([args, kwargs], sort_keys=True, default=str)
        h = hashlib.md5(string.encode("utf8")).hexdigest()
        return h + ".grib"

    def get_data(self, args, kwargs, path):
        upload_path = os.path.realpath(path + ".to_upload")
        ds = original_from_source("mars", *args, **kwargs)
        ds.save(upload_path)
        print(f"Mockup: Saving to {upload_path} for {args}, {kwargs}")
        exe = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../tools/upload-sample-dataset.py"))
        print()
        print("⚠️ To upload the test data, run this:")
        print()
        print(f"{exe} {upload_path} anemoi-datasets/create/{os.path.basename(path)} --overwrite")
        print()
        exit(1)
        raise ValueError("Test data is missing")

    def mars(self, args, kwargs):
        filename = self.filename(args, kwargs)
        dirname = "."
        path = os.path.join(dirname, filename)
        url = TEST_DATA_ROOT + "/" + filename

        assert url.startswith("http:") or url.startswith("https:")

        if not os.path.exists(path):
            print(f"Mockup: Loading url {path} for {args}, {kwargs}")
            try:
                download(url, path + ".tmp")
                os.rename(path + ".tmp", path)
            except requests.exceptions.HTTPError as e:
                print(e)
                if e.response.status_code == 404:
                    self.get_data(args, kwargs, path)
                raise

        return original_from_source("file", path)

    def __call__(self, name, *args, **kwargs):
        if name == "mars":
            return self.mars(args, kwargs)

        return original_from_source(name, *args, **kwargs)


_from_source = LoadSource()


def compare_dot_zattrs(a, b, path, errors):
    if isinstance(a, dict):
        a_keys = list(a.keys())
        b_keys = list(b.keys())
        for k in set(a_keys) | set(b_keys):
            if k not in a_keys:
                errors.append(f"❌ {path}.{k} : missing key (only in reference)")
                continue
            if k not in b_keys:
                errors.append(f"❌ {path}.{k} : additional key (missing in reference)")
                continue
            if k in [
                "timestamp",
                "uuid",
                "latest_write_timestamp",
                "history",
                "provenance",
                "provenance_load",
                "description",
                "config_path",
                "total_size",
            ]:
                if type(a[k]) is not type(b[k]):
                    errors.append(f"❌ {path}.{k} : type differs {type(a[k])} != {type(b[k])}")
                continue

            compare_dot_zattrs(a[k], b[k], f"{path}.{k}", errors)

        return

    if isinstance(a, list):
        if len(a) != len(b):
            errors.append(f"❌ {path} : lengths are different {len(a)} != {len(b)}")
            return

        for i, (v, w) in enumerate(zip(a, b)):
            compare_dot_zattrs(v, w, f"{path}.{i}", errors)

        return

    if type(a) is not type(b):
        msg = f"❌ {path} actual != expected : {a} ({type(a)}) != {b} ({type(b)})"
        errors.append(msg)
        return

    if a != b:
        msg = f"❌ {path} actual != expected : {a} != {b}"
        errors.append(msg)


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
        self.output = output_path or os.path.join(name + ".zarr")
        self.reference_path = reference_path
        print(f"Comparing {self.output} and {self.reference_path}")

        self.z_output = open_zarr(self.output)
        self.z_reference = open_zarr(self.reference_path)

        self.z_reference["data"]
        self.ds_output = open_dataset(self.output)
        self.ds_reference = open_dataset(self.reference_path)

    def compare(self):
        errors = []
        compare_dot_zattrs(dict(self.z_output.attrs), dict(self.z_reference.attrs), "metadata", errors)
        if errors:
            print("Comparison failed")
            print("\n".join(errors))

        if errors:
            raise AssertionError("Comparison failed")

        compare_datasets(self.ds_output, self.ds_reference)
        compare_statistics(self.ds_output, self.ds_reference)
        # do not compare tendencies statistics yet, as we don't know yet if they should stay


@pytest.mark.parametrize("name", NAMES)
@mockup_from_source
def test_run(name):
    config = os.path.join(HERE, name + ".yaml")
    output = os.path.join(HERE, name + ".zarr")

    creator_factory("init", config=config, path=output, overwrite=True).run()
    creator_factory("load", path=output).run()
    creator_factory("finalise", path=output).run()
    creator_factory("patch", path=output).run()
    creator_factory("init_additions", path=output, delta=["12h"]).run()
    creator_factory("run_additions", path=output, delta=["12h"]).run()
    creator_factory("finalise_additions", path=output, delta=["12h"]).run()
    creator_factory("cleanup", path=output).run()
    creator_factory("cleanup", path=output, delta=["12h"]).run()
    creator_factory("verify", path=output).run()

    # reference_path = os.path.join(HERE, name + "-reference.zarr")
    s3_uri = TEST_DATA_ROOT + "/" + name + ".zarr"
    # if not os.path.exists(reference_path):
    #    from anemoi.utils.s3 import download as s3_download
    #    s3_download(s3_uri + '/', reference_path, overwrite=True)

    Comparer(name, output_path=output, reference_path=s3_uri).compare()
    # Comparer(name, output_path=output, reference_path=reference_path).compare()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the test case")
    args = parser.parse_args()
    test_run(args.name)
