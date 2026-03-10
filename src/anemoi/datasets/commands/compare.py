# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any

import numpy as np
import tqdm
import zarr

from anemoi.datasets.create.statistics import STATISTICS
from anemoi.datasets.usage.store import dataset_lookup
from anemoi.datasets.usage.store import open_zarr

from . import Command

LOG = logging.getLogger(__name__)


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
        return f"👻 {self.message}"


class Missing(_Error):

    def __repr__(self):
        return f"❌ {self.message}"


class Error(_Error):

    def __repr__(self):
        return f"❗ {self.message}"


class Warning(_Error):

    fatal = False

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

    def warning(self, info=None):
        self._errors.append(Warning(info))

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


def _compare_arrays_partial(
    errors, a: zarr.Array, b: zarr.Array, slice_obj: slice, path: str, tolerance=1e-6, close_ok=False
) -> None:
    """Compare two arrays."""
    a = a[slice_obj]
    b = b[slice_obj]

    if np.array_equal(a, b, equal_nan=True):
        return True

    if close_ok:
        rtol = tolerance
        atol = tolerance * max(np.nanmax(np.abs(a)), np.nanmax(np.abs(b)))
        if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):

            return True

    close = "close" if np.allclose(a, b, equal_nan=True) else "different"

    errors.error(f"🧮 {path}: arrays are {close} {a[:]}, {b[:]} {a[:]-b[:]}")
    return False


def _compare_arrays(errors, a: zarr.Array, b: zarr.Array, path: str, tolerance=1e-6, close_ok=False) -> None:
    if a.shape != b.shape:
        errors.error(f"🧮 {path}: shapes are different {a.shape} != {b.shape}")
        return

    if a.dtype != b.dtype:
        errors.error(f"🧮 {path}: dtypes are different {a.dtype} != {b.dtype}")
        return

    buffer_size = 64 * 1024 * 1024  # 64 MB

    size = a.dtype.itemsize * a.size
    row_size = a.dtype.itemsize * a.shape[0]
    if size <= buffer_size:
        return _compare_arrays_partial(errors, a, b, slice(None), path, tolerance, close_ok=close_ok)

    max_workers = os.cpu_count() or 4
    max_memory = 1024 * 1024 * 1024  # 2 GB
    # Divide by two because both arrays need to be in memory
    max_workers = max(min(max_workers, max_memory // buffer_size // 2), 1)

    LOG.info(f"Comparing large arrays {path} using {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        tasks = []
        step = max(1, buffer_size // row_size)
        for i in tqdm.tqdm(range(0, a.shape[0], step), desc=f"Comparing {path}"):
            last = min(i + step, a.shape[0])
            tasks.append(executor.submit(_compare_arrays_partial, errors, a, b, slice(i, last), path, tolerance))

        with tqdm.tqdm(total=len(tasks), desc=f"Comparing {path}", unit="part") as pbar:
            for future in as_completed(tasks):
                if not future.result():
                    return False
                pbar.update(1)

        return True


def _compare_zarrs(errors, reference, actual, data, *path) -> None:
    """Compare two datasets."""

    IGNORE_VALUES = (
        []
        if data
        else [
            "zarr.data",
        ]
    )

    IGNORE_MISSINGS = [
        "zarr.sums",
        "zarr.statistics_tendencies_12h_count",
        "zarr.statistics_tendencies_12h_has_nans",
        "zarr.statistics_tendencies_12h_squares",
        "zarr.statistics_tendencies_12h_sums",
        "zarr.statistics_tendencies_6h_count",
        "zarr.statistics_tendencies_6h_has_nans",
        "zarr.statistics_tendencies_6h_squares",
        "zarr.statistics_tendencies_6h_sums",
        "zarr.statistics_tendencies_1d_count",
        "zarr.statistics_tendencies_1d_has_nans",
        "zarr.statistics_tendencies_1d_squares",
        "zarr.statistics_tendencies_1d_sums",
        "zarr.count",
        "zarr.has_nans",
        "zarr.squares",
    ]

    reference_arrays = list(reference.keys())
    actual_arrays = list(actual.keys())

    for key in sorted(set(reference_arrays) | set(actual_arrays)):

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

        if ".".join(path + (key,)) in IGNORE_VALUES:
            continue

        if isinstance(a, zarr.Group) and isinstance(b, zarr.Group):
            _compare_zarrs(errors, a, b, data, *path, key)
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

        is_statistic = key.startswith("statistics_") or key in STATISTICS

        if "stdev" in key:
            # extend tolerance for standard deviation comparisons
            _compare_arrays(errors, a, b, f"{'.'.join(path)}.{key}", tolerance=1e-5, close_ok=is_statistic)
        else:
            _compare_arrays(errors, a, b, f"{'.'.join(path)}.{key}", close_ok=is_statistic)


def _compare_dot_zattrs(errors, reference: dict, actual: dict, *path) -> None:
    """Compare the attributes of two Zarr datasets."""

    IGNORE_VALUES = [
        "metadata.provenance_load",
        "metadata.uuid",
        "metadata.total_number_of_files",
        "metadata.total_size",
        "metadata.latest_write_timestamp",
        "metadata.version",
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

    if isinstance(reference, float) and isinstance(actual, float):
        if np.isnan(reference) and np.isnan(actual):
            return

        if np.isclose(reference, actual):
            return

    if reference != actual:
        msg = f"🏷️ {'.'.join(path)} {reference=} ({type(reference)}) != {actual=} ({type(actual)})"
        errors.error(msg)


def compare_anemoi_datasets(reference, actual, data) -> None:
    """Compare the actual dataset with the reference dataset."""

    actual = open_zarr(dataset_lookup(actual))
    reference = open_zarr(dataset_lookup(reference))

    errors = ErrorCollector()

    _compare_dot_zattrs(errors, dict(reference.attrs), dict(actual.attrs), "metadata")
    _compare_zarrs(errors, reference, actual, data, "zarr")

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
        command_parser.add_argument(
            "--data",
            action="store_true",
            help="Compare data contents in addition to metadata.",
        )
        command_parser.add_argument("dataset1")
        command_parser.add_argument("dataset2")

    def run(self, args: Any) -> None:
        """Run the compare command with the provided arguments.

        Parameters
        ----------
        args : Any
            The arguments passed to the command.
        """

        if not args.data:
            LOG.warning(
                "Comparing Zarr metadata and statistics, not data contents. Use --data to compare data contents."
            )

        errors = compare_anemoi_datasets(args.dataset1, args.dataset2, data=args.data)
        if errors:
            raise RuntimeError("Datasets are different")


command = Compare
