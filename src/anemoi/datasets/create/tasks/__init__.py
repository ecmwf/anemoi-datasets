# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
from abc import ABC
from abc import abstractmethod
from typing import Any


class Task(ABC):

    def __init__(self, path, **kwargs: Any):
        self.path = path

    @abstractmethod
    def run(self) -> None:
        """Run the task."""
        pass


class Chain(Task):
    def __init__(self, tasks, **kwargs: Any):
        self.tasks = tasks
        self.kwargs = kwargs

    def run(self) -> None:
        """Run the chained tasks."""
        for cls in self.tasks:
            t = cls(**self.kwargs)
            t.run()


def task_factory(name: str, observations: bool = False, trace: str | None = None, **kwargs):

    kind = "tabular" if observations else "gridded"

    module = importlib.import_module(f".{kind}.{name}", package=__package__)

    task = getattr(module, "task")

    return task(trace=trace, **kwargs)
