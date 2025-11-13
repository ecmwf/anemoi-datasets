# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

from anemoi.datasets.create.tasks import Task


class Init(Task):
    def __init__(self, *, config: str, path: str, overwrite: bool = False, test: bool = False, **kwargs: Any):
        self.config = config
        self.path = path
        self.overwrite = overwrite
        self.test = test

    def run(self) -> None:
        print(f"Init dataset at {self.path} with config {self.config}, overwrite={self.overwrite}, test={self.test}")
        # Here would be the logic to initialize the dataset


class Load(Task):
    def __init__(self, *, path: str, parts: str | None = None, use_threads: bool = False, **kwargs: Any):
        self.path = path
        self.parts = parts
        self.use_threads = use_threads

    def run(self) -> None:
        print(f"Load data into dataset at {self.path}, parts={self.parts}, use_threads={self.use_threads}")
        # Here would be the logic to load data into the dataset


class Finalise(Task):
    def __init__(self, *, path: str, **kwargs: Any):
        self.path = path

    def run(self) -> None:
        print(f"Finalise dataset at {self.path}")
        # Here would be the logic to finalise the dataset


class InitAdditions(Task):
    def __init__(self, *, path: str, **kwargs: Any):
        self.path = path

    def run(self) -> None:
        print(f"Init additions for dataset at {self.path}")
        # Here would be the logic to initialize additions


class LoadAdditions(Task):
    def __init__(self, *, path: str, **kwargs: Any):
        self.path = path

    def run(self) -> None:
        print(f"Run additions for dataset at {self.path}")
        # Here would be the logic to run additions


class FinaliseAdditions(Task):
    def __init__(self, *, path: str, **kwargs: Any):
        self.path = path

    def run(self) -> None:
        print(f"Finalise additions for dataset at {self.path}")
        # Here would be the logic to finalise additions


class Patch(Task):
    def __init__(self, *, path: str, **kwargs: Any):
        self.path = path

    def run(self) -> None:
        print(f"Patch dataset at {self.path}")
        # Here would be the logic to patch the dataset


class Cleanup(Task):
    def __init__(self, *, path: str, **kwargs: Any):
        self.path = path

    def run(self) -> None:
        print(f"Cleanup dataset at {self.path}")
        # Here would be the logic to cleanup the dataset


class Verify(Task):
    def __init__(self, *, path: str, **kwargs: Any):
        self.path = path

    def run(self) -> None:
        print(f"Verify dataset at {self.path}")
        # Here would be the logic to verify the dataset


class TaskCreator:
    """A class to create and run dataset creation tasks."""

    def init(self, *args: Any, **kwargs: Any):
        return Init(*args, **kwargs)

    def load(self, *args: Any, **kwargs: Any):

        return Load(*args, **kwargs)

    def finalise(self, *args: Any, **kwargs: Any):
        return Finalise(*args, **kwargs)

    def init_additions(self, *args: Any, **kwargs: Any):
        return InitAdditions(*args, **kwargs)

    def load_additions(self, *args: Any, **kwargs: Any):
        return LoadAdditions(*args, **kwargs)

    def finalise_additions(self, *args: Any, **kwargs: Any):
        return FinaliseAdditions(*args, **kwargs)

    def patch(self, *args: Any, **kwargs: Any):
        return Patch(*args, **kwargs)

    def cleanup(self, *args: Any, **kwargs: Any):
        return Cleanup(*args, **kwargs)

    def verify(self, *args: Any, **kwargs: Any):

        return Verify(*args, **kwargs)
