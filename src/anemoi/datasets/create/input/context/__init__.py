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

    def __init__(self):
        self.results = {}

    def trace(self, emoji, message) -> None:

        rich.print(f"{emoji}: {message}")

    def register(self, data: Any, path: list[str]) -> Any:
        """Register data in the context.

        Parameters
        ----------
        data : Any
            Data to register.
        path : list[str]
            Path where the data should be registered.

        Returns
        -------
        Any
            Registered data.
        """
        # This is a placeholder for actual registration logic.
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
                    raise KeyError(f"Path {path} not found in results: {self.results.keys()}")

        return config

    @abstractmethod
    def empty_result(self) -> Any: ...

    @abstractmethod
    def source_argument(self, argument: Any) -> Any: ...

    @abstractmethod
    def create_result(self, data: Any) -> Any: ...
