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
from collections.abc import Iterator
from functools import cached_property
from typing import Annotated
from typing import Any
from typing import Union

from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_timedelta
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag

LOG = logging.getLogger(__name__)


def extend(x: str | list[Any] | tuple[Any, ...]) -> Iterator[datetime.datetime]:
    """Extend a date range or list of dates into individual datetime objects.

    Args:
        x (Union[str, List[Any], Tuple[Any, ...]]): A date range string or list/tuple of dates.

    Returns
    -------
    Iterator[datetime.datetime]
        An iterator of datetime objects.
    """

    if isinstance(x, (list, tuple)):
        for y in x:
            yield from extend(y)
        return

    if isinstance(x, str):
        if "/" in x:
            start, end, step = x.split("/")
            start = as_datetime(start)
            end = as_datetime(end)
            step = frequency_to_timedelta(step)
            while start <= end:
                yield start
                start += step
            return

    yield as_datetime(x)


class DatesProvider(BaseModel):

    def __iter__(self) -> Iterator[datetime.datetime]:
        """Iterate over the dates.

        Yields
        ------
        Iterator[datetime.datetime]
            An iterator of datetime objects.
        """
        yield from self.values

    def __getitem__(self, i: int) -> datetime.datetime:
        """Get a date by index.

        Args:
            i (int): Index of the date.

        Returns
        -------
        datetime.datetime
            The date at the specified index.
        """
        return self.values[i]

    def __len__(self) -> int:
        """Get the number of dates.

        Returns
        -------
        int
            The number of dates.
        """
        return len(self.values)


class StartEndDates(DatesProvider):
    start: datetime.datetime
    end: datetime.datetime
    frequency: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)] = frequency_to_timedelta("1h")
    missing: list[datetime.datetime] = Field(default_factory=list)

    @cached_property
    def values(self) -> list[datetime.datetime]:
        dates = []
        date = self.start
        while date <= self.end:
            if date not in self.missing:
                dates.append(date)
            date += self.frequency
        return dates

    def start_range(self, dates) -> datetime.datetime:
        """Used for tabular datasets grouping."""
        return dates[0]

    def end_range(self, dates) -> datetime.datetime:
        """Used for tabular datasets grouping."""
        return dates[-1] + self.frequency


class ValuesDates(DatesProvider):
    values: list[datetime.datetime]


class HindcastsDates(DatesProvider):
    hindcasts: bool = True
    start: datetime.datetime
    end: datetime.datetime
    frequency: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)] = frequency_to_timedelta("1h")
    steps: list[int] = Field(default_factory=lambda: [0])
    years: int = 20


def _dates_discriminator(config_or_model: Any) -> str:
    config = config_or_model.model_dump() if isinstance(config_or_model, BaseModel) else config_or_model

    if config.get("hindcasts", False):
        return "hindcasts"

    if "values" in config:
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
