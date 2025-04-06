# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

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
]

PUBLIC_METADATA_METHODS = [
    "arguments",
    "dtype",
    "end_date",
    "resolution",
    "source",
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
    "foo",
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
    "latitudes": {},
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
        return str(self.message)


class Failure(Error):
    emoji = "💥"


class Internal(Error):
    emoji = "💣"


class Invalid(Error):
    emoji = "❌"


class Report:

    def __init__(self):
        self.report = {}

    def success(self, name):
        self.report[name] = Success()

    def failure(self, name, message):
        self.report[name] = Failure(message)

    def internal(self, name, message):
        self.report[name] = Internal(message)

    def invalid(self, name, exception):
        self.report[name] = Invalid(exception)

    def summary(self):

        maxlen = max(len(name) for name in self.report.keys())

        for name, methods in METHODS_CATEGORIES.items():
            print()
            print(f"{name.title().replace('_', ' ')}:")
            print("-" * (len(name) + 1))

            for method in methods:
                r = self.report.get(method, Unknown())
                msg = repr(r)
                if not msg.endswith("."):
                    msg += "."
                print(f"{r.emoji} {method.ljust(maxlen)}: {msg}")

        print()


def _no_validate(report, dataset, name, result):
    LOG.warning(f"Validation for {name} not implemented. Result: {type(result)}")


def verify(report, dataset, name, kwargs=None):

    try:

        validate = globals().get(f"validate_{name}", _no_validate)

        # Check if the method is still in the Dataset class
        try:
            getattr(Dataset, name)
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
                    name, f"{name} is a callable method, not an attribute. Please update KWARGS accordingly."
                )
                return
        else:
            if kwargs is not None:
                report.internal(name, f"{name} is not callable. Please remove entry from KWARGS.")
                return

        if kwargs is not None:
            result = result(**kwargs)

        try:
            validate(report, dataset, name, result)
        except Exception as e:
            report.invalid(name, e)
            return

        report.success(name)

    except Exception as e:
        report.failure(name, e)


def verify_dataset(dataset, costly_checks=False):
    """Verify the dataset."""

    report = Report()

    if costly_checks:
        # This check is expensive as it loads the entire dataset into memory
        # so we make it optional
        default_test_indexing(dataset)

        for i, x in enumerate(dataset):
            y = dataset[i]
            assert (x == y).all(), f"Dataset indexing failed at index {i}: {x} != {y}"

    for name in METHODS:
        verify(report, dataset, name, kwargs=KWARGS.get(name))

    report.summary()


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
