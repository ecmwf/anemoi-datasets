# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from typing import Any

import numpy as np
import yaml
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets import open_dataset
from anemoi.datasets.data.stores import open_zarr


class Check:

    def __init__(
        self,
        name: str,
        config_path: str,
        dataset_path: str,
        get_test_archive: callable,
        ignore_keys=None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.get_test_archive = get_test_archive
        self.ignore_keys = ignore_keys
        self.kwargs = kwargs


class CompareToReferenceCheck(Check):
    """Class to compare datasets and their metadata."""

    def __init__(
        self,
        name: str,
        config_path: str,
        dataset_path: str,
        get_test_archive: callable,
        ignore_keys=None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Comparer instance.

        Parameters
        ----------
        output_path : str, optional
            The path to the output dataset.
        reference_path : str, optional
            The path to the reference dataset.
        """

        super().__init__(name, config_path, dataset_path, get_test_archive, ignore_keys, **kwargs)

        directory = get_test_archive(f"anemoi-datasets/create/mock-mars/{name}.zarr.tgz")
        reference = os.path.join(directory, name + ".zarr")

        self.reference_path = reference
        print(f"Comparing {self.dataset_path} and {self.reference_path}")

        self.z_output = open_zarr(self.dataset_path)
        self.z_reference = open_zarr(self.reference_path)

        self.z_reference["data"]
        self.ds_output = open_dataset(self.dataset_path)
        self.ds_reference = open_dataset(self.reference_path)
        self.ignore_keys = set(ignore_keys) if ignore_keys else set()

        self.ignore_keys.update(
            {
                "metadata.latest_write_timestamp",
                "metadata.uuid",
                "metadata.provenance_load",
                "metadata.total_size",
                "metadata.history",
                "metadata.recipe.checks",
            }
        )

    def compare_datasets(self, a: object, b: object) -> None:
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

    def compare_statistics(self, ds1: object, ds2: object) -> None:
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

    def compare_dot_zattrs(self, a: dict, b: dict, errors: list, *path) -> None:
        """Compare the attributes of two Zarr datasets."""

        name = ".".join(path)
        if name in self.ignore_keys:
            return

        if type(a) is not type(b):
            msg = f"❌ {name} type mismatch actual != expected : {a} ({type(a)}) != {b} ({type(b)})"
            errors.append(msg)
            return

        if isinstance(a, dict):
            a_keys = list(a.keys())
            b_keys = list(b.keys())
            for k in set(a_keys) | set(b_keys):

                name = ".".join(path + (k,))

                if name in self.ignore_keys:
                    continue

                if k not in a_keys:
                    errors.append(f"❌ {name} : missing key (only in reference)")
                    continue

                if k not in b_keys:
                    errors.append(f"❌ {name} : additional key (missing in reference)")
                    continue

                self.compare_dot_zattrs(a[k], b[k], errors, *path, k)

            return

        if isinstance(a, list):
            if len(a) != len(b):
                errors.append(f"❌ {path} : lengths are different {len(a)} != {len(b)}")
                return

            for i, (v, w) in enumerate(zip(a, b)):
                self.compare_dot_zattrs(v, w, errors, *path, str(i))

            return

        try:
            a, b = frequency_to_timedelta(a), frequency_to_timedelta(b)
        except Exception:
            pass

        if a != b:
            msg = f"❌ {name} actual != expected : {a} != {b}"
            errors.append(msg)

    def run(self) -> None:
        """Compare the output dataset with the reference dataset.

        Raises
        ------
        AssertionError
            If the datasets or their metadata do not match.
        """
        errors = []
        self.compare_dot_zattrs(dict(self.z_output.attrs), dict(self.z_reference.attrs), errors, "metadata")

        if errors:
            print()
            print("Comparison failed")
            print("\n".join(errors))

            print()
            print("⚠️ To update the reference data, run this:")
            print("cd " + os.path.dirname(self.dataset_path))
            base = os.path.basename(self.dataset_path)
            print(f"tar zcf {base}.tgz {base}")
            print(f"scp {base}.tgz data@anemoi.ecmwf.int:public/anemoi-datasets/create/mock-mars/")
            print()
            raise AssertionError("\n".join(errors))

        self.compare_datasets(self.ds_output, self.ds_reference)
        self.compare_statistics(self.ds_output, self.ds_reference)
        # do not compare tendencies statistics yet, as we don't know yet if they should stay


class NoneCheck(Check):

    def run(self) -> None:
        pass


def check_dataset(name: str, config_path: str, dataset_path: str, get_test_archive: callable) -> None:
    """Check the created dataset against a set of checks."""

    config = yaml.safe_load(open(config_path))
    checks = config.get("checks")

    if not checks:
        raise ValueError(f"No checks defined in {config_path}")

    for c in checks:

        check, kwargs = next(iter(c.items()))

        check = "".join(word.capitalize() for word in check.split("_")) + "Check"

        if check not in globals():
            raise ValueError(f"Check {check} not implemented")

        print(f"Running check: {check} with args: {kwargs}")
        check = globals()[check](name, config_path, dataset_path, get_test_archive, **kwargs)
        check.run()
