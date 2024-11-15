# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib

import entrypoints


def assert_is_fieldlist(obj):
    from earthkit.data.indexing.fieldlist import FieldList

    assert isinstance(obj, FieldList), type(obj)


def import_function(name, kind):

    from anemoi.transform.filters import filter_registry

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

            def proc(context, data, *args, **kwargs):
                return filter_registry.create(name, *args, **kwargs)(data)

            return proc

    raise ValueError(f"Unknown {kind} '{name}'")
