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

    requires_raw_dates: bool = False

    @abstractmethod
    def bulk_load(self, store, dates: Any) -> None:
        """Bulk load the date indexing structure with the provided dates.

        If requires_raw_dates=true, bulk_load expects a 1D array of np.datetime64 objects, with the same
        length as the dataset.

        If requires_raw_date=false, bulk_load expects an 2D array of (epoch, start_idx, length) int64,
        where 'epoch' is the date in epoch seconds, 'start_idx' is the starting index of data for that date,
        and 'length' is the number of data points for that date.

        """

    @abstractmethod
    def start_end_dates(self) -> tuple[datetime.datetime, datetime.datetime]:
        """Return the first and last dates in the dataset."""

    @abstractmethod
    def boundaries(self, start: int, end: int) -> tuple[int | None, int | None]:
        """Given start and end dates as epoch seconds,
        return the corresponding start and end indices in the dataset
        or (None, None) if no data is found in the range.
        """
