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

import numpy as np

LOG = logging.getLogger(__name__)


class WindowMetaDataBase(ABC):
    def __init__(self, owner, index) -> None:
        self.owner = owner
        self.index = index

    @property
    @abstractmethod
    def latitudes(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def longitudes(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def dates(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def time_deltas(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def reference_date(self) -> np.datetime64:
        pass

    @property
    @abstractmethod
    def reference_dates(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def boundaries(self) -> list[slice]:
        pass
