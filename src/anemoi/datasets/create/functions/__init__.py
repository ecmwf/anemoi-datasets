# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import os

import entrypoints


def assert_is_fieldlist(obj):
    from earthkit.data.indexing.fieldlist import FieldList

    assert isinstance(obj, FieldList), type(obj)


class TransformProcessor:
    def __init__(self, name, registry):
        self.name = name
        self.registry = registry

    def __call__(self, context, data, *args, **kwargs):
        transformer = self.registry.create(self.name, *args, **kwargs)
        transformer.context = context
        return transformer.forward(data)


def import_function(name, kind):

    from anemoi.transform.filters import filter_registry
    from anemoi.transform.sources import source_registry

    name = name.replace("-", "_")

    plugins = {}
    for e in entrypoints.get_group_all(f"anemoi.datasets.{kind}"):
        plugins[e.name.replace("_", "-")] = e

    if name in plugins:
        return plugins[name].load()

    try:
        module = importlib.import_module(
            f".{kind}.{name}",
            package=__name__,
        )
        return module.execute
    except ModuleNotFoundError:
        pass

    if kind == "filters":
        if filter_registry.lookup(name, return_none=True):
            return TransformProcessor(name, filter_registry)

    if kind == "sources":
        if source_registry.lookup(name, return_none=True):
            return TransformProcessor(name, source_registry)

    raise ValueError(f"Unknown {kind} '{name}'")


def _all_processors_names(kind, registry):

    for e in entrypoints.get_group_all(f"anemoi.datasets.{kind}"):
        yield e.name

    here = os.path.dirname(__file__)
    root = os.path.join(here, kind)

    for file in os.listdir(root):
        if file[0] == ".":
            continue
        if file == "__init__.py":
            continue

        if file.endswith(".py"):
            yield file[:-3]

        if os.path.isdir(os.path.join(root, file)):
            if os.path.exists(os.path.join(root, file, "__init__.py")):
                yield file

    yield from registry.names()


def _all_processors(kind, registry):
    for name in _all_processors_names(kind, registry):
        try:
            yield name, import_function(name, kind)
        except Exception:
            pass


def all_sources():
    from anemoi.transform.sources import source_registry

    return list(_all_processors("sources", source_registry))


def all_filters():
    from anemoi.transform.filters import filter_registry

    return list(_all_processors("filters", filter_registry))
