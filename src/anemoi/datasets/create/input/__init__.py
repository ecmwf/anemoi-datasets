# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING
from typing import Any

from anemoi.datasets.create.input.context.field import FieldContext

if TYPE_CHECKING:
    from anemoi.datasets.create.input.action import Recipe


class InputBuilder:
    """Builder class for creating input data from configuration and data sources."""

    def __init__(self, config: dict, data_sources: dict | list, **kwargs: Any) -> None:
        """Initialize the InputBuilder.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        data_sources : dict
            Data sources.
        **kwargs : Any
            Additional keyword arguments.
        """
        self.kwargs = kwargs
        self.config = deepcopy(config)
        self.data_sources = deepcopy(dict(data_sources=data_sources))

    @cached_property
    def action(self) -> "Recipe":
        """Returns the action object based on the configuration."""
        from .action import Recipe
        from .action import action_factory

        sources = action_factory(self.data_sources, "data_sources")
        input = action_factory(self.config, "input")

        return Recipe(input, sources)

    def select(self, argument, actor) -> Any:
        """Select data based on the group of dates.

        Parameters
        ----------
        argument : GroupOfDates
            Group of dates to select data for.
        actor: Any
            The actor performing the selection.

        Returns
        -------
        Any
            Selected data.
        """
        context = FieldContext(argument, actor, **self.kwargs)
        return context.create_result(self.action(context, argument))
