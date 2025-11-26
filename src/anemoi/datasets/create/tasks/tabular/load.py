# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from ..load import LoadTask


class Load(LoadTask):
    def __init__(self, *, path: str, parts: str | None = None, use_threads: bool = False, **kwargs: Any):
        self.path = path
        self.parts = parts
        self.use_threads = use_threads

    def run(self) -> None:
        print(f"Load data into dataset at {self.path}, parts={self.parts}, use_threads={self.use_threads}")
        # Here would be the logic to load data into the dataset


task = Load
