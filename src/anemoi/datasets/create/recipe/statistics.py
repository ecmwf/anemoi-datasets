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

import numpy as np
from pydantic import BaseModel

from anemoi.datasets.usage.misc import as_first_date
from anemoi.datasets.usage.misc import as_last_date

LOG = logging.getLogger(__name__)


class Statistics(BaseModel):
    allow_nans: bool | list[str] = False
    start: str | int | datetime.datetime | None = None
    end: str | int | datetime.datetime | None = None

    @classmethod
    def default_statistics_dates(cls, dates: list[datetime.datetime]) -> tuple[datetime.datetime, datetime.datetime]:
        """Calculate default statistics dates based on the given list of dates.

        Parameters
        ----------
        dates : list of datetime.datetime
            List of datetime objects representing dates.

        Returns
        -------
        tuple of datetime.datetime
            A tuple containing the default start and end dates.
        """

        def to_datetime(d):
            if isinstance(d, np.datetime64):
                return d.tolist()
            assert isinstance(d, datetime.datetime), d
            return d

        first = dates[0]
        last = dates[-1]

        first = to_datetime(first)
        last = to_datetime(last)

        n_years = round((last - first).total_seconds() / (365.25 * 24 * 60 * 60))

        if n_years < 10:
            # leave out 20% of the data
            k = int(len(dates) * 0.8)
            end = dates[k - 1]
            LOG.info(f"Number of years {n_years} < 10, leaving out 20%. {end=}")
            return dates[0], end

        delta = 1
        if n_years >= 20:
            delta = 3
        LOG.info(f"Number of years {n_years}, leaving out {delta} years.")
        end_year = last.year - delta

        end = max(d for d in dates if to_datetime(d).year == end_year)
        return dates[0], end

    @classmethod
    def build_statistics_dates(
        cls,
        dates: list[datetime.datetime],
        start: datetime.datetime | None,
        end: datetime.datetime | None,
    ) -> tuple[str, str]:
        """Compute the start and end dates for the statistics.

        Parameters
        ----------
        dates : list of datetime.datetime
            The list of dates.
        start : Optional[datetime.datetime]
            The start date.
        end : Optional[datetime.datetime]
            The end date.

        Returns
        -------
        tuple of str
            The start and end dates in ISO format.
        """
        # if not specified, use the default statistics dates
        default_start, default_end = Statistics.default_statistics_dates(dates)
        if start is None:
            start = default_start
        if end is None:
            end = default_end

        # in any case, adapt to the actual dates in the dataset
        start = as_first_date(start, dates)
        end = as_last_date(end, dates)

        # and convert to datetime to isoformat
        start = start.astype(datetime.datetime)
        end = end.astype(datetime.datetime)
        return (start.isoformat(), end.isoformat())
