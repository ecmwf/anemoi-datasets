# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""datetime.datetime <-> epoch conversion utilities.
We cannot use timestamp() and fromtimestamp() directly because they
depend on the local timezone.
"""

from datetime import datetime
from datetime import timezone


def date_to_epoch(date):
    """Convert UTC date/time to Unix timestamp (epoch)."""
    d = datetime(
        date.year,
        date.month,
        date.day,
        date.hour,
        date.minute,
        date.second,
        date.microsecond,
        tzinfo=timezone.utc,
    )
    return d.timestamp()


def epoch_to_date(timestamp):
    """Convert Unix timestamp (epoch) to naive UTC datetime."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)
