# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import tqdm

from anemoi.datasets.buffering import ReadAheadBuffer
from anemoi.datasets.epochs import array_to_epoch

LOG = logging.getLogger(__name__)


def validate_date_ranges(data_array, dates_ranges):

    LOG.info("Validating date ranges")
    with ReadAheadBuffer(data_array) as data:
        # Check that the number of ranges found matches the count we got during building
        offset = 0
        previous_date = None
        for i, (date, start, length) in enumerate(tqdm.tqdm(dates_ranges, desc="Validating date ranges", unit="row")):
            assert (
                length > 0
            ), f"Found non-positive range for date {date} starting at index {start} ({(date, start, length)}) [{i=}]"
            assert start == offset, f"Found non-contiguous range starting at {start}, expected {offset} [{i=}]"
            assert (
                previous_date is None or date > previous_date
            ), f"Found non-increasing date {date} after {previous_date} [{i=}]"

            chunk = data[start : start + length, :]
            date_column = array_to_epoch(chunk)
            # Check that column 0 (days since epoch) of data matches the current date
            assert np.all(
                date_column == date
            ), f"Mismatch between date range {date} and data column 0 at rows {start}:{start+length} ({date_column=}) [{i=}]"

            offset += length
            previous_date = date

        assert (
            offset == data_array.shape[0]
        ), f"Total length of ranges {offset} does not match total number of rows {np.shape[0]}"
