# (C) Copyright 2025-2026 Anemoi contributors.
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

from anemoi.transform.fields import EarthkitFieldList
from anemoi.transform.fields import FieldList

LOG = logging.getLogger(__name__)


class Context(ABC):
    """Context for building input data."""

    def __init__(self, recipe) -> None:
        from anemoi.transform.naming import create_naming

        self.recipe = recipe
        self.results = {}
        self.cache = {}
        self.use_grib_paramid = recipe.build.use_grib_paramid
        self.naming = create_naming(recipe.build.variable_naming)

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
            # It is already a result (e.g. FieldList), loaded from ${a.b.c}
            # TODO: something more elegant
            return lambda *args, **kwargs: config

        return action_factory(config, *path)

    @abstractmethod
    def create_result(self, data: Any) -> Any: ...

    def origin(self, data: Any, action: Any, action_arguments: Any) -> Any:
        """Tag the action's result with its origin.

        Called by the source and filter actions after they produce their
        result, so that the data carries the source it came from and the
        filters applied to it.

        Fields carry their origin individually, in the
        ``labels.anemoi_origin`` label. Tabular data carries a single origin
        for the whole frame, in ``DataFrame.attrs["anemoi_origin"]``
        (pandas propagates ``attrs`` through most frame operations; when a
        filter drops them, the origin is recovered from the filter's input,
        see ``Filter.combine``).

        For trajectories this is called once per ``(basetime, step)``
        retrieval, exactly as for plain gridded data: the trajectory
        structure is a property of the *dates* argument, not of the origin —
        all the fields of one variable share the same origin object
        regardless of which trajectory point they belong to.

        Parameters
        ----------
        data : Any
            The data to tag (a field list or a DataFrame).
        action : Any
            The action (source or filter) that produced the data.
        action_arguments : Any
            The argument the action was called with.

        Returns
        -------
        Any
            The tagged data.
        """

        import pandas as pd

        origin = action.origin()

        if isinstance(data, pd.DataFrame):
            previous = data.attrs.get("anemoi_origin")
            data.attrs["anemoi_origin"] = origin.combine(previous, action, action_arguments)
            return data

        result = []
        for fs in data:
            previous = fs.get("labels.anemoi_origin", default=None)
            fall_through = fs.get("labels.anemoi_fall_through", default=False)
            if fall_through:
                # The field has passed unchanged through a filter
                result.append(fs)
            else:
                anemoi_origin = origin.combine(previous, action, action_arguments)
                result.append(fs.set(**{"labels.anemoi_origin": anemoi_origin}))

        return FieldList.from_fields(result)

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

        results = list(results)  # In case it's a generator
        assert results, "join: No results to join"

        # TODO: quick hack, find a more generic way to do this

        if all(isinstance(r, (EarthkitFieldList, FieldList)) for r in results):
            # earthkit 1.0: FieldList + FieldList is element-wise arithmetic;
            # use FieldList.concat() for concatenation.
            return FieldList.concat(*results)

        # Assume it's pandas-like
        import pandas as pd

        if all(isinstance(r, pd.DataFrame) for r in results):
            # ``pd.concat`` only propagates ``attrs`` when they are identical
            # on every input; combine the frame origins explicitly.
            origins = [r.attrs["anemoi_origin"] for r in results if "anemoi_origin" in r.attrs]
            frame = pd.concat(results, ignore_index=True)
            if origins:
                from .origin import Join

                frame.attrs["anemoi_origin"] = origins[0] if len(set(origins)) == 1 else Join(origins)
            return frame

        raise TypeError(f"join: Unsupported mix of types {[type(r) for r in results]}")
