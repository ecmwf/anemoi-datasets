# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from copy import deepcopy
from typing import Any
from typing import Union

import rich

from anemoi.datasets.dates.groups import GroupOfDates

from .trace import trace_select

LOG = logging.getLogger(__name__)


class Context:
    """Context for building input data."""

    use_grib_paramid = False

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


class FieldContext(Context):
    def empty_result(self) -> Any:
        import earthkit.data as ekd

        return ekd.from_source("empty")

    def source_argument(self, argument: Any) -> Any:
        return argument.dates


class InputBuilder:
    """Builder class for creating input data from configuration and data sources."""

    def __init__(self, config: dict, data_sources: Union[dict, list], **kwargs: Any) -> None:
        """Initialize the InputBuilder.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        data_sources : Union[dict, list]
            Data sources.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.kwargs = kwargs

        config = deepcopy(config)
        if data_sources:
            config = dict(
                data_sources=dict(
                    sources=data_sources,
                    input=config,
                )
            )
        self.config = config
        self.action_path = ["input"]

    @trace_select
    def select(self, group_of_dates: GroupOfDates) -> Any:
        """Select data based on the group of dates.

        Parameters
        ----------
        group_of_dates : GroupOfDates
            Group of dates to select data for.

        Returns
        -------
        Any
            Selected data.
        """
        from .action import action_factory

        """This changes the context."""
        context = FieldContext()
        action = action_factory(self.config, "input")
        return action(context, group_of_dates)

    def _trace_select(self, group_of_dates: GroupOfDates) -> str:
        """Trace the select operation.

        Parameters
        ----------
        group_of_dates : GroupOfDates
            Group of dates to select data for.

        Returns
        -------
        str
            Trace string.
        """
        return f"InputBuilder({group_of_dates})"
