# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import rich

from anemoi.datasets.dates import DatesProvider

LOG = logging.getLogger(__name__)


class Action:
    def __init__(self, config, *path):
        self.config = config
        self.path = path


class Concat(Action):
    def __init__(self, config, *path):
        super().__init__(config, *path)

        assert isinstance(config, list), f"Value must be a dict {list}"

        self.choices = []

        for item in config:

            assert "dates" in item, f"Value must contain the key 'date' {item}"
            dates = item.pop("dates")
            filtering_dates = DatesProvider.from_config(**dates)
            action = action_factory(item)

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
    def __init__(self, config, *path):
        super().__init__(config, *path)

        assert isinstance(config, list), f"Value must be a list {config}"

        self.actions = [action_factory(item, *path, "join", str(i)) for i, item in enumerate(config)]

    def __repr__(self):
        return f"Join({self.actions})"

    def __call__(self, context, argument):
        results = context.empty_result()

        for action in self.actions:
            results += action(context, argument)

        return context.register(results, self.path)


class Pipe(Action):
    def __init__(self, config, *path):
        assert isinstance(config, list), f"Value must be a list {config}"
        super().__init__(config, *path)
        self.actions = [action_factory(item, *path, "pipe", str(i)) for i, item in enumerate(config)]

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
    def __init__(self, config, *path):
        super().__init__(config, *path, self.name)

    def __call__(self, context, argument):

        config = context.resolve(self.config)  # Substitute the ${} variables in the config

        config["_type"] = self.name  # Find a better way to do this

        source = self.create_object(config)

        rich.print(f"Executing source {self.name} from {config}")

        return context.register(self.call_object(context, source, argument), self.path)


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


KLASS = {"concat": Concat, "join": Join, "pipe": Pipe}

LEN_KLASS = len(KLASS)


def make(key, config, path):

    if LEN_KLASS == len(KLASS):

        # Load pluggins
        from anemoi.transform.filters import filter_registry as transform_filter_registry
        from anemoi.transform.sources import source_registry as transform_source_registry

        from anemoi.datasets.create.filters import filter_registry as dataset_filter_registry
        from anemoi.datasets.create.sources import source_registry as dataset_source_registry

        # Register sources, local first
        for name in dataset_source_registry.registered:
            if name not in KLASS:
                KLASS[name.replace("_", "-")] = new_source(name, DatasetSourceMixin)

        for name in transform_source_registry.registered:
            if name not in KLASS:
                KLASS[name.replace("_", "-")] = new_source(name, TransformSourceMixin)

        # Register filters, local first
        for name in dataset_filter_registry.registered:
            if name not in KLASS:
                KLASS[name.replace("_", "-")] = new_filter(name, DatasetFilterMixin)

        for name in transform_filter_registry.registered:
            if name not in KLASS:
                KLASS[name.replace("_", "-")] = new_filter(name, TransformFilterMixin)

    return KLASS[key.replace("_", "-")](config, *path)


def action_factory(data, *path):
    assert isinstance(data, dict), f"Input data must be a dictionary {data}"
    assert len(data) == 1, "Input data must contain exactly one key-value pair"

    key, value = next(iter(data.items()))
    return make(key, value, path)
