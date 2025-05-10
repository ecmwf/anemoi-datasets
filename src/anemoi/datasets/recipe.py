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


class Index:
    def __init__(self, index):
        self.name = str(index)

    def __repr__(self):
        return f"Index({self.name})"


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
        self.index = [Index(i) for i in range(len(self.steps))]

    def as_dict(self, recipe):
        if len(self.steps) == 1:
            return self.steps[0].as_dict(recipe)
        return {self.name: [s.as_dict(recipe) for s in self.steps]}

    def __repr__(self):
        return f"{self.__class__.name}({','.join([str(s) for s in self.steps])})"

    def path(self, target, result, *path):
        for i, s in enumerate(self.steps):
            s.path(target, result, *path, self, self.index[i])

    def collocated(self, a, b):
        return True


class Pipe(Chain):
    name = "pipe"


class Join(Chain):
    name = "join"


class Concat(Step):
    name = "concat"

    def __init__(self, args):
        assert isinstance(args, dict), f"Invalid argument {args}"
        self.params = args

    def __setitem__(self, key, value):
        self.params[key] = value

    def as_dict(self, recipe):

        result = []

        for k, v in sorted(self.params.items()):
            result.append({"dates": dict(start=k[0], end=k[1]), **v.as_dict(recipe)})

        return {"concat": result}

    def collocated(self, a, b):
        return a[0] is b[0]

    def path(self, target, result, *path):

        for i, (k, v) in enumerate(sorted(self.params.items())):
            v.path(target, result, *path, self, Index(i))


class Base(Step):
    def __init__(self, owner, *args, **kwargs):
        self.owner = owner
        self.name = owner.name
        self.params = {}
        for a in args:
            assert isinstance(a, dict), f"Invalid argument {a}"
            self.params.update(a)
        self.params.update(kwargs)

    def as_dict(self, recipe):

        def resolve(params, recipe):
            if isinstance(params, dict):
                return {k: resolve(v, recipe) for k, v in params.items()}

            if isinstance(params, (list, tuple)):
                return [resolve(v, recipe) for v in params]

            if isinstance(params, list):
                return [resolve(v, recipe) for v in params]

            if isinstance(params, Step):
                return recipe.resolve(self, params)

            return params

        return {self.owner.name: resolve(self.params, recipe)}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.owner.name}, {','.join([f'{k}={v}' for k, v in self.params.items()])})"

    def path(self, target, result, *path):

        if self is target:
            result.append([*path, self])


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
        self.input = Join()

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

    def dump(self):
        result = {
            "description": self.description,
            "input": self.input.as_dict(self),
        }

        print(yaml.safe_dump(result))

    def concat(self, *args, **kwargs):
        return Concat(*args, **kwargs)

    def resolve(self, source, target):
        assert isinstance(target, Source), f"Only sources can be used as template {target}"

        top = Index("input")  # So we have 'input' first in the path

        path_to_source = []
        self.input.path(source, path_to_source, top)
        if len(path_to_source) == 0:
            raise ValueError(f"Source {source} not found in recipe")
        if len(path_to_source) > 1:
            raise ValueError(f"Source {source} found in multiple locations {path_to_source}")
        path_to_source = path_to_source[0]

        path_to_target = []
        self.input.path(target, path_to_target, top)
        if len(path_to_target) == 0:
            raise ValueError(f"Target {target} not found in recipe")
        if len(path_to_target) > 1:
            raise ValueError(f"Target {target} found in multiple locations {path_to_target}")
        path_to_target = path_to_target[0]

        a = [s for s in path_to_target]
        b = [s for s in path_to_source]
        common_ancestor = None
        while a[0] is b[0]:
            common_ancestor = a[0]
            a = a[1:]
            b = b[1:]

        assert common_ancestor is not None, f"Common ancestor not found between {source} and {target}"

        if not common_ancestor.collocated(a, b):
            source = ".".join(s.name for s in path_to_source)
            target = ".".join(s.name for s in path_to_target)
            raise ValueError(
                f"Source ${{{source}}} and target ${{{target}}} are not collocated (i.e. they are not branch of a 'concat')"
            )

        target = ".".join(s.name for s in path_to_target)
        return f"${{{target}}}"


if __name__ == "__main__":

    if False:
        r = Recipe()
        r.description = "test"

        m1 = r.mars(expver="0001")
        m2 = r.mars(expver="0002")
        m3 = r.mars(expver="0003")

        r.input = (m1 + m2 + m3) | r.rename(param={"2t": "2t_0002"}) | r.rescale(tp=["mm", "m"])

        r.input += r.forcings(template=m1, param=["cos_lat", "sin_lat"])

        m0 = r.mars(expver="0000")
        c = r.concat(
            {
                ("1900", "2000"): m0,
                ("2001", "2020"): r.mars(expver="0002"),
                ("2021", "2023"): (r.mars(expver="0003") + r.forcings(template=m1, param=["cos_lat", "sin_lat"])),
            },
        )

        c[("2031", "2033")] = r.mars(expver="0005")

        r.input += c

        r.dump()
    else:
        from anemoi.datasets.create import config_to_python

        print(config_to_python("x.yaml"))
