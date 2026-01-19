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

LOG = logging.getLogger(__name__)


class Context(ABC):
    """Context for building input data."""

    def __init__(self, recipe) -> None:
        self.recipe = recipe
        self.results = {}
        self.cache = {}
        self.use_grib_paramid = recipe.build.use_grib_paramid

    def trace(self, emoji, *message) -> None:

        print(f"{emoji}: {message}")

    def register(self, data: Any, path: list[str]) -> Any:

        if not path:
            return data

        assert path[0] in ("input", "data_sources"), path

        LOG.info(f"Registering data at path: {path}")
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
                    LOG.warning(f"Path not found {path}")
                    for p in sorted(self.results):
                        LOG.info(f"   Available paths: {p}")
                    raise KeyError(f"Path {path} not found in results: {self.results.keys()}")

        return config

    def create_source(self, config: Any, *path) -> Any:
        from anemoi.datasets.create.input.action import action_factory

        if not isinstance(config, dict):
            # It is already a result (e.g. ekd.FieldList), loaded from ${a.b.c}
            # TODO: something more elegant
            return lambda *args, **kwargs: config

        return action_factory(config, *path)

    @abstractmethod
    def create_result(self, data: Any) -> Any: ...

    def join(self, results: list[Any]) -> Any:
        """Join multiple results into a single result.

        Parameters
        ----------
        results : list[Any]
            The list of results to be joined.

        Returns
        -------
        Any
            The joined result.
        """

        from functools import reduce

        import earthkit.data as ekd

        results = list(results)  # In case it's a generator
        assert results, "join: No results to join"

        # TODO: quick hack, find a more generic way to do this

        if all(isinstance(r, ekd.FieldList) for r in results):
            return reduce(lambda x, y: x + y, results)

        # Assume it's pandas-like
        import pandas as pd

        if all(isinstance(r, pd.DataFrame) for r in results):
            return pd.concat(results, ignore_index=True)

        raise TypeError(f"join: Unsupported mix of types {[type(r) for r in results]}")
