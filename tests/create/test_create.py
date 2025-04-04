# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import glob
import hashlib
import json
import logging
import os
import sys
from functools import wraps
from unittest.mock import patch

import numpy as np
import pytest
from anemoi.utils.testing import get_test_archive
from anemoi.utils.testing import get_test_data
from anemoi.utils.testing import skip_if_offline
from earthkit.data import from_source as original_from_source

from anemoi.datasets import open_dataset
from anemoi.datasets.create.testing import create_dataset
from anemoi.datasets.data.stores import open_zarr

HERE = os.path.dirname(__file__)
# find_yamls
NAMES = sorted([os.path.basename(path).split(".")[0] for path in glob.glob(os.path.join(HERE, "*.yaml"))])
SKIP = ["recentre"]
SKIP += ["accumulation"]  # test not in s3 yet
SKIP += ["regrid"]
NAMES = [name for name in NAMES if name not in SKIP]
assert NAMES, "No yaml files found in " + HERE


def mockup_from_source(func: callable) -> callable:
    """Decorator to mock the `from_source` function from the `earthkit.data` module.

    Parameters
    ----------
    func : function
        The function to be wrapped.

    Returns
    -------
    function
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with patch("earthkit.data.from_source", _from_source):
            return func(*args, **kwargs)

    return wrapper


class LoadSource:
    """Class to load data sources and handle mockup data."""

    def filename(self, args: tuple, kwargs: dict) -> str:
        """Generate a filename based on the arguments and keyword arguments.

        Parameters
        ----------
        args : tuple
            The positional arguments.
        kwargs : dict
            The keyword arguments.

        Returns
        -------
        str
            The generated filename.
        """
        string = json.dumps([args, kwargs], sort_keys=True, default=str)
        h = hashlib.md5(string.encode("utf8")).hexdigest()
        return h + ".grib"

    def get_data(self, args: tuple, kwargs: dict, path: str) -> None:
        """Retrieve data and save it to the specified path.

        Parameters
        ----------
        args : tuple
            The positional arguments.
        kwargs : dict
            The keyword arguments.
        path : str
            The path to save the data.

        Raises
        ------
        ValueError
            If the test data is missing.
        """
        upload_path = os.path.realpath(path + ".to_upload")
        ds = original_from_source("mars", *args, **kwargs)
        ds.save(upload_path)
        print(f"Mockup: Saving to {upload_path} for {args}, {kwargs}")
        print()
        print("⚠️ To upload the test data, run this:")
        path = os.path.relpath(upload_path, os.getcwd())
        name = os.path.basename(upload_path).replace(".to_upload", "")
        print(f"scp {path} data@anemoi.ecmwf.int:public/anemoi-datasets/create/mock-mars/{name}")
        print()
        exit(1)
        raise ValueError("Test data is missing")

    def mars(self, args: tuple, kwargs: dict) -> object:
        """Load data from the MARS archive.

        Parameters
        ----------
        args : tuple
            The positional arguments.
        kwargs : dict
            The keyword arguments.

        Returns
        -------
        object
            The loaded data source.
        """

        name = self.filename(args, kwargs)

        try:
            return original_from_source("file", get_test_data(f"anemoi-datasets/create/mock-mars/{name}"))
        except RuntimeError:
            raise  # If offline
        except Exception:
            self.get_data(args, kwargs, name)

    def __call__(self, name: str, *args: tuple, **kwargs: dict) -> object:
        """Call the appropriate method based on the data source name.

        Parameters
        ----------
        name : str
            The name of the data source.
        args : tuple
            The positional arguments.
        kwargs : dict
            The keyword arguments.

        Returns
        -------
        object
            The loaded data source.
        """
        if name == "mars":
            return self.mars(args, kwargs)

        return original_from_source(name, *args, **kwargs)


_from_source = LoadSource()


def compare_dot_zattrs(a: dict, b: dict, path: str, errors: list) -> None:
    """Compare the attributes of two Zarr datasets.

    Parameters
    ----------
    a : dict
        The attributes of the first dataset.
    b : dict
        The attributes of the second dataset.
    path : str
        The current path in the attribute hierarchy.
    errors : list
        The list to store error messages.
    """
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


def compare_datasets(a: object, b: object) -> None:
    """Compare two datasets.

    Parameters
    ----------
    a : object
        The first dataset.
    b : object
        The second dataset.

    Raises
    ------
    AssertionError
        If the datasets do not match.
    """
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
            abs_error = np.abs(a_ - b_)
            rel_error = np.abs(a_ - b_) / (np.abs(b_) + 1e-10)  # Avoid division by zero
            assert max_delta == 0.0, (date, param, a_, b_, a_ - b_, max_delta, np.max(abs_error), np.max(rel_error))


def compare_statistics(ds1: object, ds2: object) -> None:
    """Compare the statistics of two datasets.

    Parameters
    ----------
    ds1 : object
        The first dataset.
    ds2 : object
        The second dataset.

    Raises
    ------
    AssertionError
        If the statistics do not match.
    """
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
    """Class to compare datasets and their metadata.

    Parameters
    ----------
    name : str
        The name of the dataset.
    output_path : str, optional
        The path to the output dataset.
    reference_path : str, optional
        The path to the reference dataset.
    """

    def __init__(self, name: str, output_path: str = None, reference_path: str = None) -> None:
        """Initialize the Comparer instance.

        Parameters
        ----------
        name : str
            The name of the dataset.
        output_path : str, optional
            The path to the output dataset.
        reference_path : str, optional
            The path to the reference dataset.
        """
        self.name = name
        self.output_path = output_path or os.path.join(name + ".zarr")
        self.reference_path = reference_path
        print(f"Comparing {self.output_path} and {self.reference_path}")

        self.z_output = open_zarr(self.output_path)
        self.z_reference = open_zarr(self.reference_path)

        self.z_reference["data"]
        self.ds_output = open_dataset(self.output_path)
        self.ds_reference = open_dataset(self.reference_path)

    def compare(self) -> None:
        """Compare the output dataset with the reference dataset.

        Raises
        ------
        AssertionError
            If the datasets or their metadata do not match.
        """
        errors = []
        compare_dot_zattrs(dict(self.z_output.attrs), dict(self.z_reference.attrs), "metadata", errors)
        if errors:
            print("Comparison failed")
            print("\n".join(errors))

        if errors:
            print()

            print()
            print("⚠️ To update the reference data, run this:")
            print("cd " + os.path.dirname(self.output_path))
            base = os.path.basename(self.output_path)
            print(f"tar zcf {base}.tgz {base}")
            print(f"scp {base}.tgz data@anemoi.ecmwf.int:public/anemoi-datasets/create/mock-mars/")
            print()
            raise AssertionError("Comparison failed")

        compare_datasets(self.ds_output, self.ds_reference)
        compare_statistics(self.ds_output, self.ds_reference)
        # do not compare tendencies statistics yet, as we don't know yet if they should stay


@skip_if_offline
@pytest.mark.parametrize("name", NAMES)
@mockup_from_source
def test_run(name: str) -> None:
    """Run the test for the specified dataset.

    Parameters
    ----------
    name : str
        The name of the dataset.

    Raises
    ------
    AssertionError
        If the comparison fails.
    """
    config = os.path.join(HERE, name + ".yaml")
    output = os.path.join(HERE, name + ".zarr")
    is_test = False

    create_dataset(config=config, output=output, delta=["12h"], is_test=is_test)

    directory = get_test_archive(f"anemoi-datasets/create/mock-mars/{name}.zarr.tgz")
    reference = os.path.join(directory, name + ".zarr")

    Comparer(name, output_path=output, reference_path=reference).compare()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = NAMES

    for name in names:
        logging.info(f"Running test for {name}")
        try:
            test_run(name)
        except AssertionError:
            pass
