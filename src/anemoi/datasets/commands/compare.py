# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any

import numpy as np
import zarr

from anemoi.datasets.usage.store import dataset_lookup
from anemoi.datasets.usage.store import open_zarr

from . import Command


class _Error:
    fatal = True

    def __init__(self, message):
        self.message = message


class Added(_Error):

    fatal = False

    def __repr__(self):
        return f"🆕 {self.message}"


class MissingOK(_Error):

    fatal = False

    def __repr__(self):
        return f"⚰️ {self.message}"


class Missing(_Error):

    def __repr__(self):
        return f"❌ {self.message}"


class Error(_Error):

    def __repr__(self):
        return f"❗ {self.message}"


class ErrorCollector:
    def __init__(self):
        self._errors = []

    def added(self, info=None):
        self._errors.append(Added(info))

    def missing(self, info=None):
        self._errors.append(Missing(info))

    def missing_ok(self, info=None):
        self._errors.append(MissingOK(info))

    def error(self, info=None):
        self._errors.append(Error(info))

    def __bool__(self):
        return any(e.fatal for e in self._errors)

    def report(self):
        print()
        print("-" * 80)
        for e in self._errors:
            print(e)
        print("-" * 80)
        print()

    def __repr__(self):
        return "\n" + "\n".join(repr(e) for e in self._errors) + "\n"


def _compare_arrays(errors, a: zarr.Array, b: zarr.Array, path: str, tolerance=1e-6) -> None:
    """Compare two arrays."""
    if np.array_equal(a, b, equal_nan=True):
        return

    if a.dtype == np.dtype("float64") or b.dtype == np.dtype("float64"):
        # allows float64 -> float32 conversion errors.
        rtol = tolerance
        atol = tolerance * max(np.nanmax(np.abs(a)), np.nanmax(np.abs(b)))
        if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            return

    if np.allclose(a, b, equal_nan=True):
        errors.error(f"🧮 {path}: arrays are close but not equal {a[:]}, {b[:]}, {a[:]-b[:]}")
        return

    errors.error(f"🧮 {path}: arrays are different {a[:]}, {b[:]} {a[:]-b[:]}")


def _compare_zarrs(errors, reference, actual, *path) -> None:
    """Compare two datasets."""

    IGNORE_VALUES = []

    IGNORE_MISSINGS = [
        "zarr.sums",
        "zarr.statistics_tendencies_12h_count",
        "zarr.statistics_tendencies_12h_has_nans",
        "zarr.statistics_tendencies_12h_squares",
        "zarr.statistics_tendencies_12h_sums",
        "zarr.count",
        "zarr.has_nans",
        "zarr.squares",
    ]

    reference_arrays = list(reference.keys())
    actual_arrays = list(actual.keys())

    for key in sorted(set(reference_arrays) | set(actual_arrays)):

        if ".".join(path + (key,)) in IGNORE_VALUES:
            continue

        if key not in actual_arrays:
            if ".".join(path + (key,)) in IGNORE_MISSINGS:
                errors.missing_ok(f"🧮 {'.'.join(path)}.{key}")
            else:
                errors.missing(f"🧮 {'.'.join(path)}.{key}")
            continue

        if key not in reference_arrays:
            errors.added(f"🧮 {'.'.join(path)}.{key}")
            continue

    for key in sorted(set(reference_arrays) & set(actual_arrays)):
        a = reference[key]
        b = actual[key]

        if isinstance(a, zarr.Group) and isinstance(b, zarr.Group):
            _compare_zarrs(errors, a, b, *path, key)
            continue

        if not isinstance(a, zarr.Array) or not isinstance(b, zarr.Array):
            errors.error(f"🧮 {'.'.join(path)}.{key}: types are different {type(a)} != {type(b)}")
            continue

        if a.shape != b.shape:
            errors.error(f"🧮 {'.'.join(path)}.{key}: shapes are different {a.shape} != {b.shape}")
            continue

        if a.dtype != b.dtype:
            errors.error(f"🧮 {'.'.join(path)}.{key}: dtypes are different {a.dtype} != {b.dtype}")
            continue

        if "stdev" in key:
            # extend tolerance for standard deviation comparisons
            _compare_arrays(errors, a, b, f"{'.'.join(path)}.{key}", tolerance=1e-5)
        else:
            _compare_arrays(errors, a, b, f"{'.'.join(path)}.{key}")


def _compare_dot_zattrs(errors, reference: dict, actual: dict, *path) -> None:
    """Compare the attributes of two Zarr datasets."""

    IGNORE_VALUES = [
        "metadata.provenance_load",
        "metadata.uuid",
        "metadata.total_number_of_files",
        "metadata.total_size",
        "metadata.latest_write_timestamp",
    ]

    IGNORE_MISSINGS = [
        "metadata.history",
        "metadata.recipe.dates.group_by",
        "metadata.recipe.output.statistics",
    ]

    if type(reference) is not type(actual):
        if reference != actual:
            msg = f"🏷️ {'.'.join(path)}: {reference=} ({type(reference)}) != {actual=} ({type(actual)})"
            errors.error(msg)
        return

    if isinstance(reference, dict):
        reference_keys = set(reference.keys())
        actual_keys = set(actual.keys())
        for k in sorted(reference_keys | actual_keys):

            if k not in reference_keys:
                errors.added(f"🏷️ {'.'.join(path)}.{k}")
                continue

            if k not in actual_keys:
                if ".".join(path + (k,)) in IGNORE_MISSINGS:
                    errors.missing_ok(f"🏷️ {'.'.join(path)}.{k}")
                else:
                    errors.missing(f"🏷️ {'.'.join(path)}.{k}")
                continue

            if ".".join(path + (k,)) in IGNORE_VALUES:
                continue

            _compare_dot_zattrs(errors, reference[k], actual[k], *path, k)

        return

    if isinstance(reference, list):
        if len(reference) != len(actual):
            errors.error(f"{'.'.join(path)} lengths are different reference={len(reference)} != actual={len(actual)}")
            return

        for i, (v, w) in enumerate(zip(reference, actual)):
            _compare_dot_zattrs(errors, v, w, *path, str(i))

        return

    if reference != actual:
        msg = f"🏷️ {'.'.join(path)} {reference=} ({type(reference)}) != {actual=} ({type(actual)})"
        errors.error(msg)


def compare_anemoi_datasets(reference, actual) -> None:
    """Compare the actual dataset with the reference dataset."""

    actual = open_zarr(dataset_lookup(actual))
    reference = open_zarr(dataset_lookup(reference))

    errors = ErrorCollector()

    _compare_dot_zattrs(errors, dict(reference.attrs), dict(actual.attrs), "metadata")
    _compare_zarrs(errors, reference, actual, "zarr")

    errors.report()

    return errors


class Compare(Command):
    """Compare two datasets. This command compares the variables in two datasets and prints the mean of the common variables. It does not compare the data itself (yet)."""

    def add_arguments(self, command_parser: Any) -> None:
        """Add arguments to the command parser.

        Parameters
        ----------
        command_parser : Any
            The command parser to which arguments are added.
        """
        command_parser.add_argument("dataset1")
        command_parser.add_argument("dataset2")

    def run(self, args: Any) -> None:
        """Run the compare command with the provided arguments.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """

        errors = compare_anemoi_datasets(args.dataset1, args.dataset2)
        if errors:
            raise RuntimeError("Datasets are different")


command = Compare
