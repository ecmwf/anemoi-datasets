# (C) Copyright 2020 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import importlib
import os

import entrypoints


def assert_is_fieldlist(obj):
    from earthkit.data.indexing.fieldlist import FieldList

    assert isinstance(obj, FieldList), type(obj)


def import_function(name, kind):

    name = name.replace("-", "_")

    plugins = {}
    for e in entrypoints.get_group_all(f"anemoi.datasets.{kind}"):
        plugins[e.name.replace("_", "-")] = e

    if name in plugins:
        return plugins[name].load()

    module = importlib.import_module(
        f".{kind}.{name}",
        package=__name__,
    )
    return module.execute


def function_schemas(kind):
    plugins = {}
    for e in entrypoints.get_group_all(f"anemoi.datasets.{kind}"):
        plugins[e.name.replace("_", "-")] = e

    for name, plugin in plugins.items():
        yield name, plugin.load().schema

    path = os.path.join(os.path.dirname(__file__), kind)
    print(path)
    for n in os.listdir(path):
        print(n)
        if n.startswith("_"):
            continue

        if not n.endswith(".py"):
            continue
        name = n.replace(".py", "")
        module = importlib.import_module(
            f".{kind}.{name}",
            package=__name__,
        )

        if not hasattr(module, "schema"):
            continue

        print(module, module.schema)
        yield name, module.schema
