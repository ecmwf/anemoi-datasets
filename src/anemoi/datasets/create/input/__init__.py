# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from copy import deepcopy
from typing import Any
from typing import Union

from anemoi.datasets.create.input.context.field import FieldContext


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

    def select(self, argument) -> Any:
        """Select data based on the group of dates.

        Parameters
        ----------
        argument : GroupOfDates
            Group of dates to select data for.

        Returns
        -------
        Any
            Selected data.
        """
        from .action import action_factory

        """This changes the context."""
        context = FieldContext(argument, **self.kwargs)
        action = action_factory(self.config, "input")
        return context.create_result(action(context, argument))
