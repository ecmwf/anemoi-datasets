# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
import os
from typing import Any

import cftime
import numpy as np
from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.create.statistics import default_statistics_dates
from anemoi.datasets.use.gridded.misc import as_first_date
from anemoi.datasets.use.gridded.misc import as_last_date

LOG = logging.getLogger(__name__)


def _json_tidy(o: Any) -> Any:
    """Convert various types to JSON serializable format.

    Parameters
    ----------
    o : Any
        The object to convert.

    Returns
    -------
    Any
        The JSON serializable object.
    """
    if isinstance(o, datetime.datetime):
        return o.isoformat()

    if isinstance(o, datetime.datetime):
        return o.isoformat()

    if isinstance(o, datetime.timedelta):
        return frequency_to_string(o)

    if isinstance(o, cftime.DatetimeJulian):
        import pandas as pd

        o = pd.Timestamp(
            o.year,
            o.month,
            o.day,
            o.hour,
            o.minute,
            o.second,
        )
        return o.isoformat()

    if isinstance(o, (np.float32, np.float64)):
        return float(o)

    raise TypeError(f"{repr(o)} is not JSON serializable {type(o)}")


def _build_statistics_dates(
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
    default_start, default_end = default_statistics_dates(dates)
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


def validate_config(config: Any) -> None:

    import jsonschema

    def _tidy(d):
        if isinstance(d, dict):
            return {k: _tidy(v) for k, v in d.items()}

        if isinstance(d, list):
            return [_tidy(v) for v in d if v is not None]

        # jsonschema does not support datetime.date
        if isinstance(d, datetime.datetime):
            return d.isoformat()

        if isinstance(d, datetime.date):
            return d.isoformat()

        return d

    # https://json-schema.org

    with open(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "schemas",
            "recipe.json",
        )
    ) as f:
        schema = json.load(f)

    try:
        jsonschema.validate(instance=_tidy(config), schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        LOG.error("❌ Config validation failed (jsonschema):")
        LOG.error(e.message)
        raise
