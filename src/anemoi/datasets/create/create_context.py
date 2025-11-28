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
from abc import abstractmethod
from typing import Any

LOG = logging.getLogger(__name__)


class CreateContextBase(ABC):
    def __init__(self, path: str, config: dict, **kwargs: Any):
        self.path = path
        self.config = config
        self.kwargs = kwargs

    @abstractmethod
    def init(self): ...
    @abstractmethod
    def load(self): ...
    @abstractmethod
    def statistics(self): ...
    @abstractmethod
    def size(self): ...
    @abstractmethod
    def cleanup(self): ...

    @abstractmethod
    def init_additions(self): ...

    @abstractmethod
    def load_additions(self): ...

    @abstractmethod
    def finalise_additions(self): ...

    @abstractmethod
    def patch(self): ...

    @abstractmethod
    def verify(self): ...
