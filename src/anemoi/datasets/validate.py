# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import math
from collections import defaultdict

import numpy as np

from anemoi.datasets.data.dataset import Dataset
from anemoi.datasets.testing import default_test_indexing

LOG = logging.getLogger(__name__)
# List of methods called during training. To update the list, run training with ANEMOI_DATASETS_TRACE=1

TRAINING_METHODS = [
    "__getitem__",
    "__len__",
    "latitudes",
    "longitudes",
    "metadata",  # Accessed when checkpointing
    "missing",
    "name_to_index",
    "shape",
    "statistics",
    "supporting_arrays",  # Accessed when checkpointing
    "variables",
]

EXTRA_TRAINING_METHODS = [
    "statistics_tendencies",
]

DEBUGGING_METHODS = [
    "plot",
    "to_index",
    "tree",
    "source",
]

PUBLIC_METADATA_METHODS = [
    "arguments",
    "dtype",
    "end_date",
    "resolution",
    "start_date",
    "field_shape",
    "frequency",
    "dates",
    "typed_variables",
    "variables_metadata",
]

PRIVATE_METADATA_METHODS = [
    "computed_constant_fields",
    "constant_fields",
    "dataset_metadata",
    "label",
    "metadata_specific",
    "provenance",
]

INTERNAL_METHODS = [
    "mutate",
    "swap_with_parent",
    "dates_interval_to_indices",
]

EXPERIMENTAL_METHODS = [
    "get_dataset_names",
    "name",
    "grids",
]

OTHER_METHODS = [
    "collect_input_sources",
    "collect_supporting_arrays",
    "sub_shape",
]


METHODS_CATEGORIES = {k: v for k, v in list(globals().items()) if k.endswith("_METHODS")}


METHODS = set(sum(METHODS_CATEGORIES.values(), []))


KWARGS = {
    "__len__": {},
    "__getitem__": {"index": 0},
    "get_dataset_names": {"names": set()},
    "metadata": {},
    "metadata_specific": {},
    "mutate": {},
    "plot": {"date": 0, "variable": 0},
    "provenance": {},
    "source": {"index": 0},
    "statistics_tendencies": {},
    "sub_shape": {},
    "supporting_arrays": {},
    "swap_with_parent": {},
    "to_index": {"date": 0, "variable": 0},
    "tree": {},
}


class Unknown:
    emoji = "❓"


class Success:
    emoji = "✅"
    success = True

    def __repr__(self):
        return "Success"


class Error:
    success = False

    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return str(self.message) or repr(self.message) or "Error"


class Failure(Error):
    emoji = "💥"


class Internal(Error):
    emoji = "💣"


class Invalid(Error):
    emoji = "❌"


class Report:

    def __init__(self):
        self.report = {}
        self.methods = {}
        self.warnings = defaultdict(list)

    def method(self, name, method):
        self.methods[name] = method

    def success(self, name):
        self.report[name] = Success()

    def failure(self, name, message):
        self.report[name] = Failure(message)

    def internal(self, name, message):
        self.report[name] = Internal(message)

    def invalid(self, name, exception):
        self.report[name] = Invalid(exception)

    def warning(self, name, message):
        self.warnings[name].append(message)

    def summary(self, detailed=False):

        maxlen = max(len(name) for name in self.report.keys())

        for name, methods in METHODS_CATEGORIES.items():
            print()
            print(f"{name.title().replace('_', ' ')}:")
            print("-" * (len(name) + 1))
            print()

            for method in methods:
                r = self.report.get(method, Unknown())
                msg = repr(r)
                if not msg.endswith("."):
                    msg += "."
                print(f"{r.emoji} {method.ljust(maxlen)}: {msg}")

                for w in self.warnings.get(method, []):
                    print(" " * (maxlen + 4), "⚠️", w)

                if r.success:
                    continue

                if not detailed:
                    continue

                if method not in self.methods:
                    continue

                proc = self.methods[method]

                doc = proc.__doc__
                if doc:
                    width = 80
                    indent = maxlen + 4
                    doc = "\n".join(["=" * width, "", doc, "=" * width])
                    indented_doc = "\n".join(" " * indent + line for line in doc.splitlines())
                    print()
                    print(indented_doc)
                    print()
                    print()

        print()


def _no_validate(report, dataset, name, result):
    report.warning(name, f"Validation for {name} not implemented. Result: {type(result)}")


def validate_variables(report, dataset, name, result):
    """Validate the variables of the dataset."""

    if not isinstance(result, (list, tuple)):
        raise ValueError(f"Result is not a list or tuple {type(result)}")

    if len(result) != dataset.shape[1]:
        raise ValueError(f"Result has wrong length: {len(result)} != {dataset.shape[1]}")

    for value in result:
        if not isinstance(value, str):
            raise ValueError(f"`{value}` is not a string")


def validate_latitudes(report, dataset, name, result):
    """Validate the latitudes of the dataset."""

    if not isinstance(result, np.ndarray):
        raise ValueError(f"Result is not a np.ndarray {type(result)}")

    if len(result) != dataset.shape[3]:
        raise ValueError(f"Result has wrong length: {len(result)} != {dataset.shape[3]}")

    if not np.all(np.isfinite(result)):
        raise ValueError("Result contains non-finite values")

    if np.isnan(result).any():
        report.invalid(name, ValueError("Result contains NaN values"))
        return

    if not np.all((result >= -90) & (result <= 90)):
        raise ValueError("Result contains values outside the range [-90, 90]")

    if np.all((result >= -np.pi) & (result <= np.pi)):
        report.warning(name, "All latitudes are in the range [-π, π]. Are they in radians?")


def validate_longitudes(report, dataset, name, result):
    """Validate the longitudes of the dataset."""

    if not isinstance(result, np.ndarray):
        raise ValueError(f"Result is not a np.ndarray {type(result)}")

    if len(result) != dataset.shape[3]:
        raise ValueError(f"Result has wrong length: {len(result)} != {dataset.shape[2]}")

    if not np.all(np.isfinite(result)):
        raise ValueError("Result contains non-finite values")

    if np.isnan(result).any():
        report.invalid(name, ValueError("Result contains NaN values"))
        return

    if not np.all((result >= -180) & (result <= 360)):
        raise ValueError("Result contains values outside the range [-180, 360]")

    if np.all((result >= -np.pi) & (result <= 2 * np.pi)):
        report.warning(name, "All longitudes are in the range [-π, 2π]. Are they in radians?")


def validate_statistics(report, dataset, name, result):
    """Validate the statistics of the dataset."""

    if not isinstance(result, dict):
        raise ValueError(f"Result is not a dict {type(result)}")

    for key in ["mean", "stdev", "minimum", "maximum"]:

        if key not in result:
            raise ValueError(f"Result does not contain `{key}`")

        if not isinstance(result[key], np.ndarray):
            raise ValueError(f"Result[{key}] is not a np.ndarray {type(result[key])}")

        if len(result[key].shape) != 1:
            raise ValueError(f"Result[{key}] has wrong shape: {len(result[key].shape)} != 1")

        if result[key].shape[0] != len(dataset.variables):
            raise ValueError(f"Result[{key}] has wrong length: {result[key].shape[0]} != {len(dataset.variables)}")

        if not np.all(np.isfinite(result[key])):
            raise ValueError(f"Result[{key}] contains non-finite values")

        if np.isnan(result[key]).any():
            report.invalid(name, ValueError(f"Result[{key}] contains NaN values"))


def validate_shape(report, dataset, name, result):
    """Validate the shape of the dataset."""

    if not isinstance(result, tuple):
        raise ValueError(f"Result is not a tuple {type(result)}")

    if len(result) != 4:
        raise ValueError(f"Result has wrong length: {len(result)} != {len(dataset.shape)}")

    if result[0] != len(dataset):
        raise ValueError(f"Result[0] has wrong length: {result[0]} != {len(dataset)}")

    if result[1] != len(dataset.variables):
        raise ValueError(f"Result[1] has wrong length: {result[1]} != {len(dataset.variables)}")

    if result[2] != 1:  # We ignore ensemble dimension for now
        pass

    if result[3] != len(dataset.latitudes):
        raise ValueError(f"Result[3] has wrong length: {result[3]} != {len(dataset.latitudes)}")


def validate_supporting_arrays(report, dataset, name, result):
    """Validate the supporting arrays of the dataset."""

    if not isinstance(result, dict):
        raise ValueError(f"Result is not a dict {type(result)}")

    if "latitudes" not in result:
        raise ValueError("Result does not contain `latitudes`")

    if "longitudes" not in result:
        raise ValueError("Result does not contain `longitudes`")

    if not isinstance(result["latitudes"], np.ndarray):
        raise ValueError(f"Result[latitudes] is not a np.ndarray {type(result['latitudes'])}")

    if not isinstance(result["longitudes"], np.ndarray):
        raise ValueError(f"Result[longitudes] is not a np.ndarray {type(result['longitudes'])}")

    if np.any(result["latitudes"] != dataset.latitudes):
        raise ValueError("Result[latitudes] does not match dataset.latitudes")

    if np.any(result["longitudes"] != dataset.longitudes):
        raise ValueError("Result[longitudes] does not match dataset.longitudes")


def validate_dates(report, dataset, name, result):
    """Validate the dates of the dataset."""

    if not isinstance(result, np.ndarray):
        raise ValueError(f"Result is not a np.ndarray {type(result)}")

    if len(result.shape) != 1:
        raise ValueError(f"Result has wrong shape: {len(result.shape)} != 1")

    if result.shape[0] != len(dataset.dates):
        raise ValueError(f"Result has wrong length: {result.shape[0]} != {len(dataset.dates)}")

    if not np.issubdtype(result.dtype, np.datetime64):
        raise ValueError(f"Result is not a datetime64 array {result.dtype}")

    if len(result) != len(dataset.dates):
        raise ValueError(f"Result has wrong length: {len(result)} != {len(dataset.dates)}")

    if not np.all(np.isfinite(result)):
        raise ValueError("Result contains non-finite values")

    if np.isnan(result).any():
        report.invalid(name, ValueError("Result contains NaN values"))
        return

    for d1, d2 in zip(result[:-1], result[1:]):
        if d1 >= d2:
            raise ValueError(f"Result contains non-increasing dates: {d1} >= {d2}")

    frequency = np.diff(result)
    if not np.all(frequency == frequency[0]):
        raise ValueError("Result contains non-constant frequency")


def validate_metadata(report, dataset, name, result):
    """Validate the metadata of the dataset."""

    if not isinstance(result, dict):
        raise ValueError(f"Result is not a dict {type(result)}")


def validate_missing(report, dataset, name, result):
    """Validate the missing values of the dataset."""

    if not isinstance(result, set):
        raise ValueError(f"Result is not a set {type(result)}")

    if not all(isinstance(item, int) for item in result):
        raise ValueError("Result contains non-integer values")

    if len(result) > 0:
        if min(result) < 0:
            raise ValueError("Result contains negative values")

        if max(result) >= len(dataset):
            raise ValueError(f"Result contains values greater than {len(dataset)}")


def validate_name_to_index(report, dataset, name, result):
    """Validate the name to index mapping of the dataset."""

    if not isinstance(result, dict):
        raise ValueError(f"Result is not a dict {type(result)}")

    for key in dataset.variables:
        if key not in result:
            raise ValueError(f"Result does not contain `{key}`")

        if not isinstance(result[key], int):
            raise ValueError(f"Result[{key}] is not an int {type(result[key])}")

        if result[key] < 0 or result[key] >= len(dataset.variables):
            raise ValueError(f"Result[{key}] is out of bounds: {result[key]}")

    index_to_name = {v: k for k, v in result.items()}
    for i in range(len(dataset.variables)):
        if i not in index_to_name:
            raise ValueError(f"Result does not contain index `{i}`")

        if not isinstance(index_to_name[i], str):
            raise ValueError(f"Result[{i}] is not a string {type(index_to_name[i])}")

        if index_to_name[i] != dataset.variables[i]:
            raise ValueError(
                f"Result[{i}] does not match dataset.variables[{i}]: {index_to_name[i]} != {dataset.variables[i]}"
            )


def validate___getitem__(report, dataset, name, result):
    """Validate the __getitem__ method of the dataset."""

    if not isinstance(result, np.ndarray):
        raise ValueError(f"Result is not a np.ndarray {type(result)}")

    if result.shape != dataset.shape[1:]:
        raise ValueError(f"Result has wrong shape: {result.shape} != {dataset.shape[1:]}")


def validate___len__(report, dataset, name, result):
    """Validate the __len__ method of the dataset."""

    if not isinstance(result, int):
        raise ValueError(f"Result is not an int {type(result)}")

    if result != dataset.shape[0]:
        raise ValueError(f"Result has wrong length: {result} != {len(dataset)}")

    if result != len(dataset.dates):
        raise ValueError(f"Result has wrong length: {result} != {len(dataset.dates)}")


def validate_start_date(report, dataset, name, result):
    """Validate the start date of the dataset."""

    if not isinstance(result, np.datetime64):
        raise ValueError(f"Result is not a datetime64 {type(result)}")

    if result != dataset.dates[0]:
        raise ValueError(f"Result has wrong start date: {result} != {dataset.dates[0]}")


def validate_end_date(report, dataset, name, result):
    """Validate the end date of the dataset."""

    if not isinstance(result, np.datetime64):
        raise ValueError(f"Result is not a datetime64 {type(result)}")

    if result != dataset.dates[-1]:
        raise ValueError(f"Result has wrong end date: {result} != {dataset.dates[-1]}")


def validate_field_shape(report, dataset, name, result):
    """Validate the field shape of the dataset."""

    if not isinstance(result, tuple):
        raise ValueError(f"Result is not a tuple {type(result)}")

    if math.prod(result) != dataset.shape[-1]:
        raise ValueError(f"Result has wrong shape: {result} != {dataset.shape[-1]}")


def validate(report, dataset, name, kwargs=None):

    try:

        validate_fn = globals().get(f"validate_{name}", _no_validate)

        # Check if the method is still in the Dataset class
        try:
            report.method(name, getattr(Dataset, name))
        except AttributeError:
            report.internal(name, "Attribute not found in Dataset class. Please update the list of methods.")
            return

        # Check if the method is supported by the dataset instance
        try:
            result = getattr(dataset, name)
        except AttributeError as e:
            report.failure(name, e)
            return

        # Check if the method is callable
        if callable(result):
            if kwargs is None:
                report.internal(
                    name, f"`{name}` is a callable method, not an attribute. Please update KWARGS accordingly."
                )
                return
        else:
            if kwargs is not None:
                report.internal(name, f"`{name}` is not callable. Please remove entry from KWARGS.")
                return

        if kwargs is not None:
            result = result(**kwargs)

        if isinstance(result, np.ndarray) and np.isnan(result).any():
            report.invalid(name, ValueError("Result contains NaN values"))
            return

        try:
            validate_fn(report, dataset, name, result)
        except Exception as e:
            report.invalid(name, e)
            return

        report.success(name)

    except Exception as e:
        report.failure(name, e)


def validate_dtype(report, dataset, name, result):
    """Validate the dtype of the dataset."""

    if not isinstance(result, np.dtype):
        raise ValueError(f"Result is not a np.dtype {type(result)}")


def validate_dataset(dataset, costly_checks=False, detailed=False):
    """Validate the dataset."""

    report = Report()

    if costly_checks:
        # This check is expensive as it loads the entire dataset into memory
        # so we make it optional
        default_test_indexing(dataset)

        for i, x in enumerate(dataset):
            y = dataset[i]
            assert (x == y).all(), f"Dataset indexing failed at index {i}: {x} != {y}"

    for name in METHODS:
        validate(report, dataset, name, kwargs=KWARGS.get(name))

    report.summary(detailed=detailed)


if __name__ == "__main__":
    methods = METHODS_CATEGORIES.copy()
    methods.pop("OTHER_METHODS")

    o = set(OTHER_METHODS)
    overlap = False
    for m in methods:
        if set(methods[m]).intersection(set(OTHER_METHODS)):
            print(
                f"WARNING: {m} contains methods from OTHER_METHODS: {set(methods[m]).intersection(set(OTHER_METHODS))}"
            )
            o = o - set(methods[m])
            overlap = True

    for m in methods:
        for n in methods:
            if n is not m:
                if set(methods[m]).intersection(set(methods[n])):
                    print(
                        f"WARNING: {m} and {n} have methods in common: {set(methods[m]).intersection(set(methods[n]))}"
                    )

    if overlap:
        print(sorted(o))
