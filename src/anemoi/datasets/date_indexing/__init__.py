# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

from anemoi.utils.registry import Registry

LOG = logging.getLogger(__name__)


date_indexing_registry = Registry(__name__)


def create_date_indexing(config: Any, store: Any) -> Any:
    return date_indexing_registry.from_config(config, store=store)


class DateIndexing(ABC):

    # If true, bulk_load expects a 1D array of np.datetime64 objects
    # If false, bulk_load expects an 2D array of (epoch, start_idx, length) int64
    requires_raw_dates: bool = False

    @abstractmethod
    def bulk_load(self, store, date_ranges: Any) -> None:
        pass

    @abstractmethod
    def start_end_dates(self) -> tuple[datetime.datetime, datetime.datetime]:
        pass
