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
    def __init__(self, owner, *args, **kwargs):
        self.owner = owner
        self.args = args
        self.kwargs = kwargs

    def as_dict(self):
        return {self.owner.name: self.kwargs}


class Source(Step):
    pass


class Filter(Step):
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
            "input": [s.as_dict() for s in self._steps],
        }

        if len(result["input"]) == 1:
            result = result["input"][0]
        else:
            result["input"] = {"join": result["input"]}

        print(yaml.safe_dump(result))


if __name__ == "__main__":
    r = Recipe()
    r.description = "test"

    r.add(r.mars())
    r.add(r.rename(r.mars()))

    r.dump()
