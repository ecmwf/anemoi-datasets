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

LOG = logging.getLogger(__name__)


class Predicate:
    def __init__(self, config):
        self.config = config

    def __repr__(self):
        return f"Predicate({self.config})"

    def match(self, dates):
        # Just a demo
        raise NotImplementedError("Not yet implemented")
        return True


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
            predicate = Predicate(item.pop("dates"))
            action = action_factory(item)

            self.choices.append((predicate, action))

    def __repr__(self):
        return f"Concat({self.choices})"

    def __call__(self, context, argument):

        for predicate, action in self.choices:
            if predicate.match(argument):
                return context.register(
                    action(context, argument),
                    self.path,
                )

        raise ValueError(f"No matching predicate for dates: {argument}")


class Join(Action):
    def __init__(self, config, *path):
        super().__init__(config, *path)

        assert isinstance(config, list), f"Value must be a list {config}"

        self.actions = [
            action_factory(
                item,
                *path,
                "join",
                str(i),
            )
            for i, item in enumerate(config)
        ]

    def __repr__(self):
        return f"Join({self.actions})"

    def __call__(self, context, argument):
        results = context.empty_result()
        for action in self.actions:
            results += action(context, argument)
        return context.register(
            results,
            self.path,
        )


class Pipe(Action):
    def __init__(self, config, *path):
        assert isinstance(config, list), f"Value must be a list {config}"
        super().__init__(config, *path)
        self.actions = [
            action_factory(
                item,
                *path,
                "pipe",
                str(i),
            )
            for i, item in enumerate(config)
        ]

    def __repr__(self):
        return f"Pipe({self.actions})"

    def __call__(self, context, argument):
        result = context.empty_result()

        for i, action in enumerate(self.actions):
            if i == 0:
                result = action(context, argument)
            else:
                result = action(context, result)

        return context.register(
            result,
            self.path,
        )


class Function(Action):
    def __init__(self, config, *path):
        super().__init__(config, *path, self.name)


class SourceFunction(Function):

    def __call__(self, context, argument):
        from anemoi.datasets.create.sources import create_source

        config = context.resolve(self.config)  # Substitute the ${} variables in the config
        config["_type"] = self.name  # Find a better way to do this
        source = create_source(self, config)

        rich.print(f"Executing source {self.name} from {config}")

        return context.register(
            source.execute(context, context.source_argument(argument)),
            self.path,
        )


class FilterFunction(Function):
    pass


def new_source(name):
    return type(name.title(), (SourceFunction,), {"name": name})


def new_filter(name):
    return type(name.title(), (FilterFunction,), {"name": name})


KLASS = {"concat": Concat, "join": Join, "pipe": Pipe}

LEN_KLASS = len(KLASS)


def make(key, config, path):

    if LEN_KLASS == len(KLASS):
        # Load pluggins
        from anemoi.datasets.create.sources import registered_sources

        for name in registered_sources():
            assert name not in KLASS, f"Duplicate source name: {name}"
            KLASS[name] = new_source(name)

    return KLASS[key](config, *path)


def action_factory(data, *path):
    assert isinstance(data, dict), f"Input data must be a dictionary {data}"
    assert len(data) == 1, "Input data must contain exactly one key-value pair"

    key, value = next(iter(data.items()))
    return make(key, value, path)
