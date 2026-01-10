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

    @abstractmethod
    def bulk_load(self, store, dates_ranges: Any) -> None:
        """Bulk load the date indexing structure with the provided dates.

        bulk_load expects an 2D array of (epoch, start_idx, length) int64,
        where 'epoch' is the date in epoch seconds, 'start_idx' is the starting index of data for that date,
        and 'length' is the number of data points for that date.

        """

    @abstractmethod
    def start_end_dates(self) -> tuple[datetime.datetime, datetime.datetime]:
        """Return the first and last dates in the dataset."""

    @abstractmethod
    def range_search(self, start: int, end: int, dataset_length: int) -> slice:
        """Given start and end dates as epoch seconds,
        return the corresponding slice (start and end are included).
        """

    @classmethod
    def validate_bulk_load_input(cls, dates_ranges: Any, data_length: int) -> None:
        """Validate the input for bulk loading the date indexing structure.

        Parameters
        ----------
        dates_ranges : Any
            The input data to validate.

        data_length : int
            The expected total length of the data.

        Raises
        ------
        ValueError
            If the input data does not meet the expected format or type.
        """
        import numpy as np
        import tqdm

        if not isinstance(dates_ranges, (np.ndarray,)):
            raise ValueError("dates_ranges must be a numpy ndarray")

        if dates_ranges.ndim != 2 or dates_ranges.shape[1] != 3:
            raise ValueError("dates_ranges must be a 2D array with shape (N, 3)")

        if dates_ranges.dtype != np.int64:
            raise ValueError("dates_ranges must have dtype of int64")

        assert isinstance(data_length, int), type(data_length)

        # Validate
        total = 0
        last = None
        for i, (epoch, offset, length) in enumerate(tqdm.tqdm(dates_ranges, desc="Validating date ranges")):
            if last is None:
                last = epoch
            else:
                assert epoch > last, (epoch, last)
                last = epoch

            assert 0 <= offset < data_length, (offset, data_length, i)
            assert length > 0, (length, i)

            assert offset == total, (offset, total, i)
            assert offset < data_length, (offset, data_length, i)

            total += length

        if total != data_length:
            raise ValueError(f"Total length {total} does not match expected data length {data_length}")
