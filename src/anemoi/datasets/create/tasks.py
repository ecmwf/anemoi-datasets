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


class Task(ABC):
    @abstractmethod
    def run(self) -> None:
        """Run the task."""
        pass


def chain(tasks: list) -> type:
    """Create a class to chain multiple tasks.

    Parameters
    ----------
    tasks : list
        The list of tasks to chain.

    Returns
    -------
    type
        The class to chain multiple tasks.
    """

    class Chain(Task):
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

        def run(self) -> None:
            """Run the chained tasks."""
            for cls in tasks:
                t = cls(**self.kwargs)
                t.run()

    return Chain


def task_factory(name: str, trace: str | None = None, **kwargs):

    if True:
        from anemoi.datasets.create.gridded.tasks import TaskCreator

        creator = TaskCreator()
    else:
        from anemoi.datasets.create.observations.tasks import TaskCreator

        creator = TaskCreator()

    return getattr(creator, name)(trace=trace, **kwargs)
