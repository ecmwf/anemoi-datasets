# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.datasets.dates import DatesProvider

LOG = logging.getLogger(__name__)


class Action:
    """An "Action" represents a single operation described in the yaml configuration, e.g. a source, a filter,
    pipe, join, etc.

    See :ref:`operations` for more details.

    """

    def __init__(self, config, *path):
        self.config = config
        self.path = path
        assert path[0] in (
            "input",
            "data_sources",
        ), f"{self.__class__.__name__}: path must start with 'input' or 'data_sources': {path}"


class Concat(Action):
    """The Concat contruct is used to concat different actions that are responsible
    for delivery fields for different dates.

    See :ref:`building-concat` for more details.

    .. block-code:: yaml

        input:
            concat:
                - dates:
                    start: 2023-01-01
                    end: 2023-01-31
                    frequency: 1d
                  action: # some action
                     ...

                - dates:
                    start: 2023-02-01
                    end: 2023-02-28
                    frequency: 1d
                  action: # some action

    """

    def __init__(self, config, *path):
        super().__init__(config, *path, "concat")

        assert isinstance(config, list), f"Value must be a dict {list}"

        self.choices = []

        for i, item in enumerate(config):

            dates = item["dates"]
            filtering_dates = DatesProvider.from_config(**dates)
            action = action_factory({k: v for k, v in item.items() if k != "dates"}, *self.path, str(i))
            self.choices.append((filtering_dates, action))

    def __repr__(self):
        return f"Concat({self.choices})"

    def __call__(self, context, argument):

        results = context.empty_result()

        for filtering_dates, action in self.choices:
            dates = context.matching_dates(filtering_dates, argument)
            if len(dates) == 0:
                continue
            results += action(context, dates)

        return context.register(results, self.path)


class Join(Action):
    """Implement the join operation to combine results from multiple actions.

    See :ref:`building-join` for more details.

    .. block-code:: yaml

        input:
            join:
                - grib:
                     ...

                - netcdf: # some other action
                     ...

    """

    def __init__(self, config, *path):
        super().__init__(config, *path, "join")

        assert isinstance(config, list), f"Value of Join Action must be a list, got: {config}"

        self.actions = [action_factory(item, *self.path, str(i)) for i, item in enumerate(config)]

    def __repr__(self):
        return f"Join({self.actions})"

    def __call__(self, context, argument):
        results = context.empty_result()

        for action in self.actions:
            results += action(context, argument)

        return context.register(results, self.path)


class Pipe(Action):
    """Implement the pipe operation to chain results from a
    source through multiple filters.

    See :ref:`building-pipe` for more details.

    .. block-code:: yaml

        input:
            pipe:
                - grib:
                     ...

                - rename:
                     ...

    """

    def __init__(self, config, *path):
        assert isinstance(config, list), f"Value of Pipe Action must be a list, got {config}"
        super().__init__(config, *path, "pipe")
        self.actions = [action_factory(item, *self.path, str(i)) for i, item in enumerate(config)]

    def __repr__(self):
        return f"Pipe({self.actions})"

    def __call__(self, context, argument):
        result = context.empty_result()

        for i, action in enumerate(self.actions):
            if i == 0:
                result = action(context, argument)
            else:
                result = action(context, result)

        return context.register(result, self.path)


class Function(Action):
    """Base class for sources and filters."""

    def __init__(self, config, *path):
        super().__init__(config, *path, self.name)

    def __call__(self, context, argument):

        config = context.resolve(self.config)  # Substitute the ${} variables in the config

        config["_type"] = self.name  # Find a better way to do this

        source = self.create_object(context, config)

        return context.register(self.call_object(context, source, argument), self.path)


class DatasetSourceMixin:
    """Mixin class for sources defined in anemoi-datasets"""

    def create_object(self, context, config):
        from anemoi.datasets.create.sources import create_source as create_datasets_source

        return create_datasets_source(context, config)

    def call_object(self, context, source, argument):
        return source.execute(context.source_argument(argument))


class TransformSourceMixin:
    """Mixin class for sources defined in anemoi-transform"""

    def create_object(self, context, config):
        from anemoi.transform.sources import create_source as create_transform_source

        return create_transform_source(context, config)


class TransformFilterMixin:
    """Mixin class for filters defined in anemoi-transform"""

    def create_object(self, context, config):
        from anemoi.transform.filters import create_filter as create_transform_filter

        return create_transform_filter(context, config)

    def call_object(self, context, filter, argument):
        return filter.forward(context.filter_argument(argument))


class FilterFunction(Function):
    """Action to call a filter on the argument (e.g. rename, regrid, etc.)."""

    def __call__(self, context, argument):
        return self.call(context, argument, context.filter_argument)


def _make_name(name, what):
    name = name.replace("_", "-")
    name = "".join(x.title() for x in name.split("-"))
    return name + what.title()


def new_source(name, mixin):
    return type(
        _make_name(name, "source"),
        (Function, mixin),
        {"name": name},
    )


def new_filter(name, mixin):
    return type(
        _make_name(name, "filter"),
        (Function, mixin),
        {"name": name},
    )


class DataSources(Action):
    """Action to call a source (e.g. mars, netcdf, grib, etc.)."""

    def __init__(self, config, *path):
        super().__init__(config, *path)
        assert isinstance(config, (dict, list)), f"Invalid config type: {type(config)}"
        if isinstance(config, dict):
            self.sources = {k: action_factory(v, *path, k) for k, v in config.items()}
        else:
            self.sources = {i: action_factory(v, *path, str(i)) for i, v in enumerate(config)}

    def __call__(self, context, argument):
        for name, source in self.sources.items():
            context.register(source(context, argument), self.path + (name,))


class Recipe(Action):
    """Action that represent a recipe (i.e. a sequence of data_sources and input)."""

    def __init__(self, input, data_sources):
        self.input = input
        self.data_sources = data_sources

    def __call__(self, context, argument):
        # Load data_sources
        self.data_sources(context, argument)
        return self.input(context, argument)


KLASS = {
    "concat": Concat,
    "join": Join,
    "pipe": Pipe,
    "data-sources": DataSources,
}

LEN_KLASS = len(KLASS)


def make(key, config, *path):

    if LEN_KLASS == len(KLASS):

        # Load pluggins
        from anemoi.transform.filters import filter_registry as transform_filter_registry
        from anemoi.transform.sources import source_registry as transform_source_registry

        from anemoi.datasets.create.sources import source_registry as dataset_source_registry

        # Register sources, local first
        for name in dataset_source_registry.registered:
            if name not in KLASS:
                KLASS[name.replace("_", "-")] = new_source(name, DatasetSourceMixin)

        for name in transform_source_registry.registered:
            if name not in KLASS:
                KLASS[name.replace("_", "-")] = new_source(name, TransformSourceMixin)

        # Register filters
        for name in transform_filter_registry.registered:
            if name not in KLASS:
                KLASS[name.replace("_", "-")] = new_filter(name, TransformFilterMixin)

    return KLASS[key.replace("_", "-")](config, *path)


def action_factory(data, *path):

    assert len(path) > 0, f"Path must contain at least one element {path}"
    assert path[0] in ("input", "data_sources")

    assert isinstance(data, dict), f"Input data must be a dictionary, got {type(data)}"
    assert len(data) == 1, f"Input data must contain exactly one key-value pair {data} {'.'.join(x for x in path)}"

    key, value = next(iter(data.items()))
    return make(key, value, *path)
