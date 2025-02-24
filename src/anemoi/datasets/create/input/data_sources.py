# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from earthkit.data import FieldList

from ...dates.groups import GroupOfDates
from .action import Action
from .action import action_factory
from .misc import _tidy
from .result import Result

LOG = logging.getLogger(__name__)


class DataSourcesAction(Action):
    """Class to handle data sources actions in the dataset creation process."""

    def __init__(
        self,
        context: object,
        action_path: List[str],
        sources: Union[Dict[str, Any], List[Dict[str, Any]]],
        input: Dict[str, Any],
    ) -> None:
        """Initializes a DataSourcesAction instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : List[str]
            The action path.
        sources : Union[Dict[str, Any], List[Dict[str, Any]]]
            The sources configuration.
        input : Dict[str, Any]
            The input configuration.
        """
        super().__init__(context, ["data_sources"], *sources)
        if isinstance(sources, dict):
            configs = [(str(k), c) for k, c in sources.items()]
        elif isinstance(sources, list):
            configs = [(str(i), c) for i, c in enumerate(sources)]
        else:
            raise ValueError(f"Invalid data_sources, expecting list or dict, got {type(sources)}: {sources}")

        self.sources = [action_factory(config, context, ["data_sources"] + [a_path]) for a_path, config in configs]
        self.input = action_factory(input, context, ["input"])

    def select(self, group_of_dates: GroupOfDates) -> "DataSourcesResult":
        """Selects the data sources result for the given group of dates.

        Parameters
        ----------
        group_of_dates : GroupOfDates
            The group of dates.

        Returns
        -------
        DataSourcesResult
            The data sources result.
        """
        sources_results = [a.select(group_of_dates) for a in self.sources]
        return DataSourcesResult(
            self.context,
            self.action_path,
            group_of_dates,
            self.input.select(group_of_dates),
            sources_results,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the DataSourcesAction instance."""
        content = "\n".join([str(i) for i in self.sources])
        return self._repr(content)


class DataSourcesResult(Result):
    """Class to represent the result of data sources actions in the dataset creation process."""

    def __init__(
        self,
        context: object,
        action_path: List[str],
        dates: object,
        input_result: Result,
        sources_results: List[Result],
    ) -> None:
        """Initializes a DataSourcesResult instance.

        Parameters
        ----------
        context : object
            The context object.
        action_path : List[str]
            The action path.
        dates : object
            The dates object.
        input_result : Result
            The input result.
        sources_results : List[Result]
            The list of sources results.
        """
        super().__init__(context, action_path, dates)
        # result is the main input result
        self.input_result = input_result
        # sources_results is the list of the sources_results
        self.sources_results = sources_results

    @cached_property
    def datasource(self) -> FieldList:
        """Returns the combined datasource from all sources."""
        for i in self.sources_results:
            # for each result trigger the datasource to be computed
            # and saved in context
            self.context.notify_result(i.action_path[:-1], i.datasource)
        # then return the input result
        # which can use the datasources of the included results
        return _tidy(self.input_result.datasource)
