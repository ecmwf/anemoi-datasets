# (C) Copyright 2025 Anemoi contributors.
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
from typing import Any

import rich

LOG = logging.getLogger(__name__)


class Context(ABC):
    """Context for building input data."""

    def __init__(self, /, argument: Any) -> None:
        self.results = {}
        self.cache = {}
        self.argument = argument

    def trace(self, emoji, *message) -> None:

        rich.print(f"{emoji}: {message}")

    def register(self, data: Any, path: list[str]) -> Any:

        if not path:
            return data

        assert path[0] in ("input", "data_sources"), path

        rich.print(f"Registering data at path: {path}")
        self.results[tuple(path)] = data
        return data

    def resolve(self, config):
        config = config.copy()

        for key, value in list(config.items()):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                path = tuple(value[2:-1].split("."))
                if path in self.results:
                    config[key] = self.results[path]
                else:
                    rich.print(f"Path not found {path}")
                    for p in sorted(self.results):
                        rich.print(f"   Available paths: {p}")
                    raise KeyError(f"Path {path} not found in results: {self.results.keys()}")

        return config

    def create_source(self, config: Any) -> Any:
        from anemoi.datasets.create.input.action import action_factory

        return action_factory(config)

    @abstractmethod
    def empty_result(self) -> Any: ...

    @abstractmethod
    def create_result(self, data: Any) -> Any: ...
