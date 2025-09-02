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

    def __init__(self, when="dataset-create"):
        self.when = when

    def __eq__(self, other):
        if not isinstance(other, Origin):
            return False
        return self is other

    def __hash__(self):
        return id(self)


def _un_dotdict(x):
    if isinstance(x, dict):
        return {k: _un_dotdict(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_un_dotdict(a) for a in x]

    return x


class Pipe(Origin):
    def __init__(self, s1, s2, when="dataset-create"):
        super().__init__(when)
        self.steps = [s1, s2]

        if isinstance(s1, Pipe):
            assert not isinstance(s2, Pipe), (s1, s2)
            self.steps = s1.steps + [s2]

    def combine(self, previous):
        assert False, (self, previous)

    def as_dict(self):
        return {
            "type": "pipe",
            "steps": [s.as_dict() for s in self.steps],
            "when": self.when,
        }

    def __repr__(self):
        return " | ".join(repr(s) for s in self.steps)


class Source(Origin):
    def __init__(self, name, config, when="dataset-create"):
        super().__init__(when)
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)

    def combine(self, previous):
        assert previous is None, f"Cannot combine origins, previous already exists: {previous}"
        return self

    def as_dict(self):
        return {
            "type": "source",
            "name": self.name,
            "config": self.config,
            "when": self.when,
        }

    def __repr__(self):
        return f"{self.name}({id(self)})"


class Filter(Origin):
    def __init__(self, name, config, when="dataset-create"):
        super().__init__(when)
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)
        self._cache = {}

    def combine(self, previous):
        if previous in self._cache:
            # We use a cache to avoid recomputing the same combination
            return self._cache[previous]
        self._cache[previous] = Pipe(previous, self)
        return self._cache[previous]

    def as_dict(self):
        return {
            "type": "filter",
            "name": self.name,
            "config": self.config,
            "when": self.when,
        }

    def __repr__(self):
        return f"{self.name}({id(self)})"
