# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np


class Task(ABC):

    def __init__(self, /, path, config, overwrite=False, cache=None, **kwargs: Any):
        # Catch all floating point errors, including overflow, sqrt(<0), etc

        np.seterr(all="raise", under="warn")

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


class TaskCreator:
    """A class to create and run dataset creation tasks."""

    def __init__(self, context, **kwargs: Any):
        self.context = context
        self.kwargs = kwargs

    def init(self):
        return self.context.init()

    def load(self):
        return self.context.load()

    def size(self):
        return self.context.size()

    def patch(self):
        return self.context.patch()

    def statistics(self):
        return self.context.statistics()

    def finalise(self):
        self.context.statistics()
        self.context.size()
        self.context.cleanup()

    def cleanup(self):
        self.context.cleanup()

    def verify(self):
        self.context.verify()

    def init_additions(self):
        self.context.init_additions()

    def load_additions(self):
        self.context.load_additions()

    def finalise_additions(self):
        self.context.finalise_additions()
        self.context.size()

    def additions(self):
        self.context.init_additions()
        self.context.load_additions()
        self.context.finalise_additions()
        self.context.size()
        self.context.cleanup()


def task_factory(name: str, observations: bool = False, trace: str | None = None, **kwargs):

    if observations:
        from anemoi.datasets.create.tabular.create_context import TabularCreateContext

        context = TabularCreateContext(**kwargs)
    else:
        from anemoi.datasets.create.gridded.create_context import GriddedCreateContext

        context = GriddedCreateContext(**kwargs)

    dispatch = TaskCreator(context)
    return getattr(dispatch, name)()
