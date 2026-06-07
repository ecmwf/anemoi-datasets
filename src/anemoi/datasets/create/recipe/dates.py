# (C) Copyright 2026- Anemoi contributors.
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
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import model_validator

LOG = logging.getLogger(__name__)


def _extend(x: str | list[Any] | tuple[Any, ...]) -> Iterator[datetime.datetime]:
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
            yield from _extend(y)
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

    class MissingRange(BaseModel):
        start: datetime.datetime
        end: datetime.datetime
        frequency: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)] | None = None

    start: datetime.datetime
    end: datetime.datetime
    frequency: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)] = frequency_to_timedelta("1h")
    missing: list[datetime.datetime | str | MissingRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def _expand_missing_ranges(self) -> "StartEndDates":
        expanded = []
        for item in self.missing:
            if isinstance(item, self.MissingRange):
                current = item.start
                step = item.frequency or self.frequency
                while current <= item.end:
                    expanded.append(current)
                    current += step
            else:
                expanded.append(as_datetime(item))

        # Keep deterministic ordering for comparisons and filtering.
        self.missing = sorted(set(expanded))
        return self

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


class BaseDates(StartEndDates):
    """Basetimes (forecast initialisation times) for the ``trajectories`` layout.

    Mirrors :class:`StartEndDates`: it models a flat list of base dates via
    ``start`` / ``end`` / ``frequency`` (plus optional ``missing``).  It is kept
    as a distinct type so the ``base_dates`` recipe field is self-documenting and
    can be exported to the JSON schema independently of analysis ``dates``.

    Unlike :class:`StartEndDates`, ``values`` retains the slots for ``missing``
    base dates: the on-disk trajectory array must keep an entry for every base
    date in the range, and :class:`~anemoi.datasets.dates.groups.TrajectoryFilter`
    removes the missing ones only from the iteration that drives data loading.
    """

    @cached_property
    def values(self) -> list[datetime.datetime]:
        dates = []
        date = self.start
        while date <= self.end:
            dates.append(date)
            date += self.frequency
        return dates


class Steps(BaseModel):
    """Forecast lead times for the ``trajectories`` layout.

    Models a regular range of steps via ``start`` / ``end`` / ``frequency``
    (all parsed as :class:`datetime.timedelta`).  Exposes the same iteration and
    ``numpy`` array interface the trajectories pipeline relies on.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)]
    end: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)]
    frequency: Annotated[datetime.timedelta, BeforeValidator(frequency_to_timedelta)]

    @cached_property
    def values(self):
        import numpy as np

        return np.arange(self.start, self.end + self.frequency, self.frequency)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __array__(self, dtype=None, copy=None):
        arr = self.values if dtype is None else self.values.astype(dtype)
        return arr.copy() if copy else arr


class TrajectoryDates(DatesProvider):
    """Dates provider for the ``trajectories`` layout.

    ``values`` is the list of ``(basetime, step)`` pairs formed by the Cartesian
    product of the ``base_dates`` provider with the forecast ``steps``.  Use
    :meth:`factorise` to recover the underlying sorted-unique basetimes and
    steps.  Missing handling is delegated to the basetimes provider; pairs are
    not masked individually.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_dates: BaseDates
    steps: Steps

    @cached_property
    def values(self) -> list[tuple[datetime.datetime, Any]]:
        return [(bt, st) for bt in self.base_dates.values for st in self.steps.values]

    @cached_property
    def missing(self) -> list[datetime.datetime]:
        return list(self.base_dates.missing)

    def factorise(self) -> tuple[list[datetime.datetime], Any]:
        """Return ``(basetimes, steps)`` as sorted-unique collections.

        Returns
        -------
        basetimes : list of datetime.datetime
            Sorted, unique basetimes extracted from ``values``.
        steps : numpy.ndarray
            Sorted, unique steps (as ``numpy.timedelta64``) extracted from
            ``values``.
        """
        import numpy as np

        basetimes = sorted({bt for bt, _ in self.values})
        steps = np.array(sorted({st for _, st in self.values}))
        return basetimes, steps

    @property
    def frequency(self) -> datetime.timedelta:
        """Frequency of the underlying basetimes provider."""
        return self.base_dates.frequency

    def __repr__(self) -> str:
        bt0, st0 = self.values[0]
        btN, stN = self.values[-1]
        return f"{self.__class__.__name__}(basetimes={bt0}..{btN}, steps={st0}..{stN}, pairs={len(self.values)})"


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
