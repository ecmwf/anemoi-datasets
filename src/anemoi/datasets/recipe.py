# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import yaml
from anemoi.transform.filters import filter_registry as transform_filter_registry

from anemoi.datasets.create.filters import filter_registry as datasets_filter_registry
from anemoi.datasets.create.sources import source_registry

LOG = logging.getLogger(__name__)


class Step:

    def __or__(self, other):
        return Pipe(self, other)

    def __add__(self, other):
        return Join(self, other)


class Chain(Step):
    def __init__(self, *args):
        if len(args) > 0 and isinstance(args[0], self.__class__):
            args = args[0].steps + args[1:]

        self.steps = args

    def as_dict(self):
        if len(self.steps) == 1:
            return self.steps[0].as_dict()
        return {self.name: [s.as_dict() for s in self.steps]}

    def __repr__(self):
        return f"{self.__class__.name}({','.join([str(s) for s in self.steps])})"


class Pipe(Chain):
    name = "pipe"


class Join(Chain):
    name = "join"


class Base(Step):
    def __init__(self, owner, *args, **kwargs):
        self.owner = owner
        self.params = {}
        for a in args:
            assert isinstance(a, dict), f"Invalid argument {a}"
            self.params.update(a)
        self.params.update(kwargs)

    def as_dict(self):
        return {self.owner.name: self.params}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.owner.name}, {','.join([f'{k}={v}' for k, v in self.params.items()])})"


class Source(Base):
    pass


class Filter(Base):
    pass


class SourceMaker:
    def __init__(self, name, factory):
        self.name = name
        self.factory = factory

    def __call__(self, *args, **kwargs):
        return Source(self, *args, **kwargs)


class FilterMaker:
    def __init__(self, name, factory):
        self.name = name
        self.factory = factory

    def __call__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], Step):
            prev = args[0]
            args = args[1:]
            return Pipe(prev, Filter(self, *args, **kwargs))
        return Filter(self, *args, **kwargs)


class Recipe:

    def __init__(self):
        self.description = None
        self._steps = []

        sources = source_registry.factories.copy()
        filters = transform_filter_registry.factories.copy()

        for key, factory in datasets_filter_registry.factories.items():
            if key in filters:
                LOG.warning(
                    f"Filter `{key}` is registered in anemoi.datasets filter registry and in anemoi.transform filter registry"
                )
            filters[key] = factory

        for key, factory in sources.items():
            if key in filters:
                LOG.warning(
                    f"Source `{key}` is registered in anemoi.datasets source registry and in anemoi.transform filter registry"
                )
                del filters[key]

        for key, factory in sources.items():
            key = key.replace("-", "_")
            assert not hasattr(self, key)
            setattr(self, key, SourceMaker(key, factory))

        for key, factory in filters.items():
            key = key.replace("-", "_")
            assert not hasattr(self, key)
            setattr(self, key, FilterMaker(key, factory))

    def add(self, step):
        self._steps.append(step)

    def dump(self):
        result = {
            "description": self.description,
            "input": Join(*self._steps).as_dict(),
        }

        print(yaml.safe_dump(result))


if __name__ == "__main__":
    r = Recipe()
    r.description = "test"

    # r.add(
    #     r.mars(
    #         expver="0001",
    #         levtype="sfc",
    #         param=["2t"],
    #         number=[0, 1],
    #     )
    # )

    # r.add(
    #     r.rescale(
    #         r.rename(
    #             r.mars(
    #                 expver="0002",
    #                 levtype="sfc",
    #                 param=["2t"],
    #                 number=[0, 1],
    #             ),
    #             param={"2t": "2t_0002"},
    #         ),
    #         {"2t_0002": ["mm", "m"]},
    #     )
    # )

    m1 = r.mars(expver="0001", levtype="sfc", param=["2t"], number=[0, 1])
    m2 = r.mars(expver="0002", levtype="sfc", param=["2t"], number=[0, 1])

    m3 = r.mars(expver="0003", levtype="sfc", param=["2t"], number=[0, 1])

    r.add(
        (m1 + m2 + m3)
        | r.rename(
            param={"2t": "2t_0002"},
        )
        | r.rescale(
            {"2t_0002": ["mm", "m"]},
        )
    )

    r.dump()
