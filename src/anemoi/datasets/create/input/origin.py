# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC

LOG = logging.getLogger(__name__)


class Origin(ABC):

    def __init__(self):
        self._variables = set()

    def __repr__(self):
        return repr(self.as_dict())

    def __eq__(self, other):
        if not isinstance(other, Origin):
            return False
        return self is other  # or self.as_dict() == other.as_dict()

    def __hash__(self):
        return id(self)

    def add_variable(self, name):
        self._variables.add(name)


def _un_dotdict(x):
    if isinstance(x, dict):
        return {k: _un_dotdict(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_un_dotdict(a) for a in x]

    return x


class Source(Origin):
    def __init__(self, name, config):
        super().__init__()
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)

    def combine(self, previous):
        assert previous is None, f"Cannot combine origins, previous already exists: {previous}"
        return self

    def as_dict(self):
        return {"type": "source", "name": self.name, "config": self.config, "variables": sorted(self._variables)}


class Filter(Origin):
    def __init__(self, name, config, previous=None):
        super().__init__()
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)
        self.previous = previous

    def combine(self, previous):
        if self.previous is previous:
            # Avoid duplication of intermediate origins
            return self
        return Filter(self.name, self.config, previous)

    def as_dict(self):
        return {
            "type": "filter",
            "name": self.name,
            "config": self.config,
            "apply_to": self.previous.as_dict(),
            "variables": sorted(self._variables),
        }
