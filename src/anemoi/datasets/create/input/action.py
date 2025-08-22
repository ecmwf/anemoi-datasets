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
    def __init__(self, config, *path):
        self.config = config
        self.path = path
        assert path[0] in (
            "input",
            "data_sources",
        ), f"{self.__class__.__name__}: path must start with 'input' or 'data_sources': {path}"


class Concat(Action):
    def __init__(self, config, *path):
        super().__init__(config, *path, "concat")

        assert isinstance(config, list), f"Value must be a dict {list}"

        self.choices = []

        for i, item in enumerate(config):

            assert "dates" in item, f"Value must contain the key 'dates' {item}"
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

    def python_code(self, code):
        return code.concat(
            {filtering_dates.to_python(): action.python_code(code) for filtering_dates, action in self.choices}
        )


class Join(Action):
    def __init__(self, config, *path):
        super().__init__(config, *path, "join")

        assert isinstance(config, list), f"Value must be a list {config}"

        self.actions = [action_factory(item, *self.path, str(i)) for i, item in enumerate(config)]

    def __repr__(self):
        return f"Join({self.actions})"

    def __call__(self, context, argument):
        results = context.empty_result()

        for action in self.actions:
            results += action(context, argument)

        return context.register(results, self.path)

    def python_code(self, code) -> None:
        return code.sum(a.python_code(code) for a in self.actions)


class Pipe(Action):
    def __init__(self, config, *path):
        assert isinstance(config, list), f"Value must be a list {config}"
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

    def python_code(self, code) -> None:
        return code.pipe(a.python_code(code) for a in self.actions)


class Function(Action):
    def __init__(self, config, *path):
        super().__init__(config, *path, self.name)

    def __call__(self, context, argument):

        config = context.resolve(self.config)  # Substitute the ${} variables in the config

        config["_type"] = self.name  # Find a better way to do this

        source = self.create_object(config)

        return context.register(self.call_object(context, source, argument), self.path)

    def python_code(self, code) -> str:
        # For now...
        if "source" in self.config:
            source = action_factory(self.config["source"], *self.path, "source")
            self.config["source"] = source.python_code(code)
        return code.call(self.name, self.config)


class DatasetSourceMixin:
    def create_object(self, config):
        from anemoi.datasets.create.sources import create_source as create_datasets_source

        return create_datasets_source(self, config)

    def call_object(self, context, source, argument):
        return source.execute(context, context.source_argument(argument))


class DatasetFilterMixin:
    def create_object(self, config):
        from anemoi.datasets.create.filters import create_filter as create_datasets_filter

        return create_datasets_filter(self, config)

    def call_object(self, context, filter, argument):
        return filter.execute(context.filter_argument(argument))


class TransformSourceMixin:
    def create_object(self, config):
        from anemoi.transform.sources import create_source as create_transform_source

        return create_transform_source(self, config)


class TransformFilterMixin:
    def create_object(self, config):
        from anemoi.transform.filters import create_filter as create_transform_filter

        return create_transform_filter(self, config)

    def call_object(self, context, filter, argument):
        return filter.forward(context.filter_argument(argument))


class FilterFunction(Function):
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
    def __init__(self, config, *path):
        super().__init__(config, *path)
        self.sources = {k: action_factory(v, *path, k) for k, v in config.items()}

    def python_code(self, code):
        return code.sources({k: v.python_code(code) for k, v in self.sources.items()})

    def __call__(self, context, argument):
        for name, source in self.sources.items():
            context.register(source(context, argument), self.path + (name,))


class Recipe(Action):
    def __init__(self, input, data_sources):
        self.input = input
        self.data_sources = data_sources

    def python_code(self, code):
        return code.recipe(
            self.input.python_code(code),
            self.data_sources.python_code(code),
        )

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
