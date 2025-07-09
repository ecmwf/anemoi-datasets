# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

LOG = logging.getLogger(__name__)


class Predicate:
    def __init__(self, config):
        self.config = config

    def __repr__(self):
        return f"Predicate({self.config})"

    def match(self, dates):
        # Just a demo
        return True


class Concat:
    def __init__(self, config):
        assert isinstance(config, list), f"Value must be a dict {list}"

        self.choices = []

        for item in config:

            assert "dates" in item, f"Value must contain the key 'date' {item}"
            predicate = Predicate(item.pop("dates"))
            action = action_factory(item)

            self.choices.append((predicate, action))

    def __repr__(self):
        return f"Concat({self.choices})"

    def __call__(self, context, group_of_dates):

        for predicate, action in self.choices:
            if predicate.match(group_of_dates):
                return action(group_of_dates)

        raise ValueError(f"No matching predicate for dates: {group_of_dates}")


class Join:
    def __init__(self, config):
        assert isinstance(config, list), f"Value must be a list {config}"
        self.actions = [action_factory(item) for item in config]

    def __repr__(self):
        return f"Join({self.actions})"

    def __call__(self, context, group_of_dates):
        results = []
        for action in self.actions:
            results.append(action(context, group_of_dates))
        return results


class Pipe:
    def __init__(self, config):
        assert isinstance(config, list), f"Value must be a list {config}"
        self.actions = [action_factory(item) for item in config]

    def __repr__(self):
        return f"Pipe({self.actions})"

    def __call__(self, context, dates):
        result = None
        for action in self.actions:
            if result is None:
                result = action(dates)
            else:
                result = action(result)
        return result


class Function:
    def __init__(self, config):
        self.config = config

    #     # if self._source:
    #     #     self.source = action_factory(config[self._source])
    #     # else:
    #     #     self.source = None

    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.config})"

    # def __call__(self, context, dates):
    #     # Just a demo, in real case it would do something with the dates

    #     config = self.config.copy()
    #     if self.source:
    #         config[self._source] = self.source(dates)

    #     return {self.__class__.__name__: self.config, "dates": dates}


class SourceFunction(Function):
    def __init__(self, config):
        from anemoi.datasets.create.sources import create_source

        super().__init__(config)
        config["_type"] = self.name
        self.source = create_source(self, config)

    def __call__(self, context, group_of_dates):
        return self.source.execute(context, group_of_dates.dates)


class FilterFunction(Function):
    pass


def new_source(name, source=None):
    return type(name.title(), (SourceFunction,), {"name": name})


def new_filter(name, source=None):
    return type(name.title(), (FilterFunction,), {"name": name})


KLASS = {
    "concat": Concat,
    "join": Join,
    "pipe": Pipe,
}

LEN_KLASS = len(KLASS)


def make(key, config):

    if LEN_KLASS == len(KLASS):
        # Load pluggins
        from anemoi.datasets.create.sources import registered_sources

        for name in registered_sources():
            assert name not in KLASS, f"Duplicate source name: {name}"
            KLASS[name] = new_source(name)

    return KLASS[key](config)


def action_factory(data, *path):
    assert isinstance(data, dict), f"Input data must be a dictionary {data}"
    assert len(data) == 1, "Input data must contain exactly one key-value pair"

    key, value = next(iter(data.items()))
    return make(key, value)
