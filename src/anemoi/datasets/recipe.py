# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import sys
from collections import defaultdict
from tempfile import TemporaryDirectory

from anemoi.transform.filters import filter_registry as transform_filter_registry
from anemoi.utils.config import DotDict
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

# from anemoi.datasets.create.filters import filter_registry as datasets_filter_registry
from anemoi.datasets.create.sources import source_registry

LOG = logging.getLogger(__name__)


def _un_dotdict(x):
    if isinstance(x, dict):
        return {k: _un_dotdict(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_un_dotdict(a) for a in x]

    return x


class Index:
    def __init__(self, index):
        self.name = str(index)

    def __repr__(self):
        return f"Index({self.name})"

    def same(self, other):
        if not isinstance(other, Index):
            return False
        return self.name == other.name


class Step:

    def __or__(self, other):
        return Pipe(self, other)

    def __and__(self, other):
        return Join(self, other)

    def same(self, other):
        return self is other


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

    def path(self, target, result, *path):

        if target is self:
            result.append([*path, self])
            return

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

            key = dict(start=as_datetime(k[0]), end=as_datetime(k[1]))
            if len(k) == 3:
                key["frequency"] = k[2]

            result.append({"dates": key, **v.as_dict(recipe)})

        return {"concat": result}

    def collocated(self, a, b):
        return a[0].same(b[0])

    def path(self, target, result, *path):
        if target is self:
            result.append([*path, self])
            return
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

        def resolve(params, recipe, name=None):
            if isinstance(params, dict):

                def _(k):
                    if isinstance(k, str) and k.endswith("_"):
                        return k[:-1]
                    return k

                return {_(k): resolve(v, recipe, name=_(k)) for k, v in params.items()}

            if isinstance(params, (list, tuple)):
                return [resolve(v, recipe) for v in params]

            if isinstance(params, list):
                return [resolve(v, recipe) for v in params]

            if isinstance(params, Step):
                return recipe.resolve(self, params, name=name)

            return params

        return {self.owner.name: resolve(self.params, recipe)}

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

    def __init__(self, name=None, description=None, attribution=None, licence=None):

        self._description = description
        self._attribution = attribution
        self._licence = licence
        self._name = name
        self._dates = None
        self._statistics = None
        self._build = None
        self._env = None
        self._dataset_status = None
        self._output = None
        self._platform = None

        self.input = Join()
        self.output = DotDict()
        self.statistics = DotDict()
        self.build = DotDict()

        self._data_sources = {}
        self._counter = defaultdict(int)

        sources = source_registry.factories.copy()
        filters = transform_filter_registry.factories.copy()

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

        self.repeated_dates = SourceMaker("repeated_dates", None)

    def as_dict(self):
        result = {
            "name": self.name,
            "description": self.description,
            "attribution": self.attribution,
            "licence": self.licence,
            "dates": self.dates,
            "statistics": self.statistics,
            "build": self.build,
        }

        if self._data_sources:
            result["data_sources"] = self._data_sources

        for k, v in list(result.items()):
            if v is None:
                del result[k]

        return result

    def concat(self, *args, **kwargs):
        return Concat(*args, **kwargs)

    def make_data_source(self, name, target):

        target = target.as_dict(self)

        name = name or "source"
        if name in self._data_sources:
            if self._data_sources[name] == target:
                return f"${{data_sources.{name}}}"

        n = self._counter[name]
        self._counter[name] += 1

        name = f"{name}_{n}" if n > 0 else name

        self._data_sources[name] = target.copy()
        return f"${{data_sources.{name}}}"

    def resolve(self, source, target, name=None):

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
        if len(path_to_target) > 1:
            raise ValueError(f"Target {target} found in multiple locations {path_to_target}")

        if len(path_to_target) == 0:
            # Add a `data_sources` entry
            return self.make_data_source(name, target)

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

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value.strip()

    @property
    def attribution(self):
        return self._attribution

    @attribution.setter
    def attribution(self, value):
        self._attribution = value.strip()

    @property
    def licence(self):
        return self._licence

    @licence.setter
    def licence(self, value):
        self._licence = value.strip()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value.strip()

    @property
    def dates(self):
        return self._dates

    def _parse_dates(self, value):

        if isinstance(value, dict):
            return value

        start = None
        end = None
        frequency = 1

        if isinstance(value, (list, tuple)):
            if len(value) in [2, 3]:
                start = value[0]
                end = value[1]

            if len(value) == 3:
                frequency = frequency_to_string(frequency_to_timedelta(value[2]))
                if isinstance(frequency, int):
                    frequency = f"{frequency}h"

        if start is None or end is None:
            raise ValueError(f"Invalid dates {value}")

        if isinstance(frequency, int):
            frequency = f"{frequency}h"

        return dict(
            start=as_datetime(start),
            end=as_datetime(end),
            frequency=frequency,
        )

    @dates.setter
    def dates(self, value):
        self._dates = self._parse_dates(value)

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def statistics(self):
        return self._statistics

    @statistics.setter
    def statistics(self, value):
        self._statistics = value

    @property
    def build(self):
        return self._build

    @build.setter
    def build(self, value):
        self._build = value

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value

    @property
    def dataset_status(self):
        return self._dataset_status

    @dataset_status.setter
    def dataset_status(self, value):
        self._dataset_status = value

    @property
    def platform(self):
        return self._platform

    @platform.setter
    def platform(self, value):
        self._platform = value

    def dump(self, file=sys.stdout):
        input = self.input.as_dict(self)  # First so we get the data_sources

        result = self.as_dict()

        result["input"] = input

        if self.output:
            result["output"] = self.output

        if self.statistics:
            result["statistics"] = self.statistics

        if self.build:
            result["build"] = self.build

        if self.env:
            result["env"] = self.env

        if self.dataset_status:
            result["dataset_status"] = self.dataset_status

        if self.platform:
            result["platform"] = self.platform

        from anemoi.datasets.dumper import yaml_dump

        yaml_dump(_un_dotdict(result), stream=file)

    def test(self, output="recipe.zarr"):
        from argparse import ArgumentParser

        from anemoi.datasets.commands.create import command

        parser = ArgumentParser()
        parser.add_argument("command", help="Command to run")

        cmd = command()
        cmd.add_arguments(parser)

        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "recipe.yaml")
            with open(path, "w") as file:
                self.dump(file)

            args = parser.parse_args(["create", path, output, "--overwrite", "--test"])
            cmd.run(args)


if __name__ == "__main__":

    r = Recipe()
    r.description = "test"

    r.dates = ("2023-01-01 00:00:00", "2023-12-31 18:00:00", "6h")

    m1 = r.mars(expver="0001", grid=[20, 20])
    m2 = r.mars(expver="0002")
    m3 = r.mars(expver="0003")

    r.input = m1

    r.input += r.forcings(template=m1, param=["cos_latitude", "sin_latitude"])

    # m0 = r.mars(expver="0000")
    # c = r.concat(
    #     {
    #         ("190", "2000"): m0,
    #         ("2001", "2020"): r.mars(expver="0002"),
    #         ("2021", "2023"): (r.mars(expver="0003") + r.forcings(template=m1, param=["cos_lat", "sin_lat"])),
    #     },
    # )

    # c[("2031", "2033")] = r.mars(expver="0005")

    # r.input += c

    r.output.group_by = "day"
    r.build.additions = True
    r.statistics.end = "80%"

    r.dump()
    r.test()
