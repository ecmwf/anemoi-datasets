# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from anemoi.utils.dates import DateTimes
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.hindcasts import HindcastDatesTimes
from anemoi.utils.humanize import print_dates
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic_core import PydanticCustomError

LOG = logging.getLogger(__name__)


class DatesProvider(BaseModel):
    pass


class StartEndDates(DatesProvider):
    start: datetime.datetime
    end: datetime.datetime
    frequency: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)] = frequency_to_timedelta("1h")
    missing: list[datetime.datetime] = Field(default_factory=list)


class ValuesDates(DatesProvider):
    values: list[datetime.datetime]


class HindcastsDates(DatesProvider):
    hindcasts: bool = True
    start: datetime.datetime
    end: datetime.datetime
    frequency: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)] = frequency_to_timedelta("1h")
    steps: list[int] = Field(default_factory=lambda: [0])
    years: int = 20


def _dates_discriminator(options: Any) -> str:

    if options.get("hindcasts", False):
        return "hindcasts"

    if "values" in options:
        return "values"

    return "start_end"


Dates = Annotated[
    Union[
        Annotated[StartEndDates, Tag("start_end")],
        Annotated[ValuesDates, Tag("values")],
        Annotated[HindcastsDates, Tag("hindcasts")],
    ],
    Discriminator(_dates_discriminator),
]
