# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from anemoi.datasets.epochs import epoch_to_date

LOG = logging.getLogger(__name__)


class DateRange:
    """Helper class to represent a date range with start and end dates."""

    def __init__(self, epoch, offset, length) -> None:
        self.epoch = epoch
        self.offset = offset
        self.length = length

    def __repr__(self):
        date = epoch_to_date(self.epoch)
        return f"DateRange(epoch={date}, offset={self.offset:,}, length={self.length:,})"

    def empty(self) -> bool:
        return self.length == 0
