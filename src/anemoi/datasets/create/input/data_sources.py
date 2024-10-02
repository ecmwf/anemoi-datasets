# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import logging
from functools import cached_property

from anemoi.utils.dates import as_datetime as as_datetime
from anemoi.utils.dates import frequency_to_timedelta as frequency_to_timedelta

from anemoi.datasets.dates import DatesProvider as DatesProvider
from anemoi.datasets.fields import FieldArray as FieldArray
from anemoi.datasets.fields import NewValidDateTimeField as NewValidDateTimeField

from .action import Action
from .action import action_factory
from .misc import _tidy
from .result import Result

LOG = logging.getLogger(__name__)


class DataSourcesAction(Action):
    def __init__(self, context, action_path, sources, input):
        super().__init__(context, ["data_sources"], *sources)
        if isinstance(sources, dict):
            configs = [(str(k), c) for k, c in sources.items()]
        elif isinstance(sources, list):
            configs = [(str(i), c) for i, c in enumerate(sources)]
        else:
            raise ValueError(f"Invalid data_sources, expecting list or dict, got {type(sources)}: {sources}")

        self.sources = [action_factory(config, context, ["data_sources"] + [a_path]) for a_path, config in configs]
        self.input = action_factory(input, context, ["input"])

    def select(self, group_of_dates):
        sources_results = [a.select(group_of_dates) for a in self.sources]
        return DataSourcesResult(
            self.context,
            self.action_path,
            group_of_dates,
            self.input.select(group_of_dates),
            sources_results,
        )

    def __repr__(self):
        content = "\n".join([str(i) for i in self.sources])
        return super().__repr__(content)


class DataSourcesResult(Result):
    def __init__(self, context, action_path, dates, input_result, sources_results):
        super().__init__(context, action_path, dates)
        # result is the main input result
        self.input_result = input_result
        # sources_results is the list of the sources_results
        self.sources_results = sources_results

    @cached_property
    def datasource(self):
        for i in self.sources_results:
            # for each result trigger the datasource to be computed
            # and saved in context
            self.context.notify_result(i.action_path[:-1], i.datasource)
        # then return the input result
        # which can use the datasources of the included results
        return _tidy(self.input_result.datasource)
