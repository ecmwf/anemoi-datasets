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

        assert s1 is not None, (s1, s2)
        assert s2 is not None, (s1, s2)

        if isinstance(s1, Pipe):
            assert not isinstance(s2, Pipe), (s1, s2)
            self.steps = s1.steps + [s2]

    def combine(self, previous, action, action_arguments):
        assert False, (self, previous)

    def as_dict(self):
        return {
            "type": "pipe",
            "steps": [s.as_dict() for s in self.steps],
            "when": self.when,
        }

    def __repr__(self):
        return " | ".join(repr(s) for s in self.steps)


class Join(Origin):
    def __init__(self, origins, when="dataset-create"):
        assert isinstance(origins, (list, tuple, set)), origins
        super().__init__(when)
        self.steps = list(origins)

        assert all(o is not None for o in origins), origins

    def combine(self, previous, action, action_arguments):
        assert False, (self, previous)

    def as_dict(self):
        return {
            "type": "join",
            "steps": [s.as_dict() for s in self.steps],
            "when": self.when,
        }

    def __repr__(self):
        return " & ".join(repr(s) for s in self.steps)


class Source(Origin):
    def __init__(self, name, config, when="dataset-create"):
        super().__init__(when)
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)

    def combine(self, previous, action, action_arguments):
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

    def combine(self, previous, action, action_arguments):

        if previous is None:
            # This can happen if the filter does not tag its output with an origin
            # (e.g. a user plugin). In that case we try to get the origin from the action arguments
            key = (id(action), id(action_arguments))
            if key not in self._cache:

                LOG.warning(f"No previous origin to combine with: {self}. Action: {action}")
                LOG.warning(f"Connecting to action arguments {action_arguments}")
                origins = set()
                for k in action_arguments:
                    o = k.metadata("anemoi_origin", default=None)
                    if o is None:
                        raise ValueError(
                            f"Cannot combine origins, previous is None and action_arguments {action_arguments} has no origin"
                        )
                    origins.add(o)
                if len(origins) == 1:
                    self._cache[key] = origins.pop()
                else:
                    self._cache[key] = Join(origins)
            previous = self._cache[key]

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
