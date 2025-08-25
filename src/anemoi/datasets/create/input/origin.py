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
    pass


def _un_dotdict(x):
    if isinstance(x, dict):
        return {k: _un_dotdict(v) for k, v in x.items()}

    if isinstance(x, (list, tuple, set)):
        return [_un_dotdict(a) for a in x]

    return x


class Source(Origin):
    def __init__(self, name, config):
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)

    def combine(self, previous):
        assert previous is None, f"Cannot combine origins, previous already exists: {previous}"
        return self

    def __repr__(self):
        return f"Source(name={self.name}, config={self.config})"


class Filter(Origin):
    def __init__(self, name, config, previous=None):
        assert isinstance(config, dict), f"Config must be a dictionary {config}"
        self.name = name
        self.config = _un_dotdict(config)
        self.previous = previous

    def combine(self, previous):
        if self.previous is previous:
            # Avoid duplication of intermediate origins
            return self
        return Filter(self.name, self.config, previous)

    def __repr__(self):
        return f"Filter(name={self.name}, config={self.config}, previous={self.previous})"
