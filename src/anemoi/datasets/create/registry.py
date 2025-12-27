# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod


class Registry(ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def get_flags(self): ...

    @abstractmethod
    def get_flag(self, index: int) -> bool: ...

    @abstractmethod
    def set_flag(self, index: int): ...

    @abstractmethod
    def add_provenance(self, name: str): ...
