# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from . import Task


class InitTask(Task):
    def __init__(self, *, config: str, path: str, overwrite: bool = False, test: bool = False, **kwargs: Any):
        self.config = config
        self.path = path
        self.overwrite = overwrite
        self.test = test

    def run(self) -> None:
        print(f"Init dataset at {self.path} with config {self.config}, overwrite={self.overwrite}, test={self.test}")
        # Here would be the logic to initialize the dataset
