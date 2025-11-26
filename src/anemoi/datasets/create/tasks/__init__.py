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

    def __init__(self, /, path, config, overwrite=False, cache=None, **kwargs: Any):
        self.path = path
        self.cache = cache
        self.config = config
        self.overwrite = overwrite

    def run(self):
        from anemoi.datasets.create.utils import cache_context

        with cache_context(self.cache):
            return self._run()

    @abstractmethod
    def _run(self) -> None: ...


class Chain(Task):
    def __init__(self, tasks, **kwargs: Any):
        self.tasks = tasks
        self.kwargs = kwargs

    def _run(self) -> None:
        """Run the chained tasks."""
        for cls in self.tasks:
            t = cls(**self.kwargs)
            t._run()


def task_factory(name: str, observations: bool = False, trace: str | None = None, **kwargs):

    kind = "tabular" if observations else "gridded"

    module = importlib.import_module(f".{kind}.{name}", package=__package__)

    task = getattr(module, "task")

    print(module.__file__)

    return task(trace=trace, **kwargs)
