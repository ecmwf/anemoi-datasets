# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from abc import abstractmethod

from anemoi.datasets import open_dataset

LOG = logging.getLogger(__name__)


class Validator(ABC):

    @abstractmethod
    def validate(self, path, ds, errors): ...

    def __repr__(self):
        return self.__class__.__name__


class UnknownValidator(Validator):
    def validate(self, path, ds, errors):
        errors.append(f"Unknown layout: {ds.layout}")


class GriddedValidator(Validator):
    def validate(self, path, ds, errors):
        # Check shape againts dates
        start, end, frequency = ds.start_date, ds.end_date, ds.frequency
        size = ds.data.shape[0]
        expected_size = ((end - start) // frequency) + 1
        if size != expected_size:
            errors.append(
                f"Gridded dataset size {size} does not match expected size {expected_size} based on dates (missing={len(ds.missing)})."
            )


class TabularValidator(Validator):
    def validate(self, path, ds, errors):
        LOG.warning(f"Tabular layout validation is not fully implemented for dataset at path: {path}")


class TrajectoriesValidator(Validator):
    def validate(self, path, ds, errors):
        LOG.warning(f"Trajectories layout validation is not fully implemented for dataset at path: {path}")


VALIDATORS = {
    "gridded": GriddedValidator(),
    "tabular": TabularValidator(),
    "trajectories": TrajectoriesValidator(),
}


def dataset_validation(path, raise_error=False):
    ds = open_dataset(path)

    validator = VALIDATORS.get(ds.layout, UnknownValidator)

    errors = []
    validator.validate(path, ds, errors)
    for error in errors:
        LOG.error(f"{validator}: {error}")

    if errors:
        if raise_error:
            raise ValueError(f"Validation errors for dataset at path {path}: {errors}")
        return False
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python validation.py <dataset_path>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    try:
        dataset_validation(dataset_path)
        print(f"Dataset at path {dataset_path} is valid.")
    except ValueError as e:
        print(e)
        sys.exit(1)
