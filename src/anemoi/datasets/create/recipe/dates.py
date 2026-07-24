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
from pydantic import ConfigDict
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import model_validator

# Frequency and Steps live in the import-light `time_schemas` module (they
# are shared with per-source validation schemas); re-exported here so
# existing `recipe.dates` imports keep working.
from anemoi.datasets.create.time_schemas import Frequency  # noqa: F401
from anemoi.datasets.create.time_schemas import Steps  # noqa: F401
from anemoi.datasets.create.time_schemas import matches_date_pattern

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
        frequency: Frequency | None = None

    start: datetime.datetime
    end: datetime.datetime
    frequency: Frequency = frequency_to_timedelta("1h")
    missing: list[datetime.datetime | str | MissingRange] = Field(default_factory=list)

    @model_validator(mode="after")
    def _expand_missing_ranges(self) -> "StartEndDates":
        expanded = []
        patterns = []
        for item in self.missing:
            if isinstance(item, self.MissingRange):
                current = item.start
                step = item.frequency or self.frequency
                while current <= item.end:
                    expanded.append(current)
                    current += step
            elif isinstance(item, str) and "?" in item:
                # A wildcard pattern selects the matching dates from the grid;
                # it is resolved below, once against the full range.
                patterns.append(item)
            else:
                expanded.append(as_datetime(item))

        for pattern in patterns:
            matched = [d for d in self._date_grid() if matches_date_pattern(d, pattern)]
            if not matched:
                LOG.warning("'missing' pattern %r matched no date in the range; ignoring it.", pattern)
            expanded.extend(matched)

        # Keep deterministic ordering for comparisons and filtering.
        self.missing = sorted(set(expanded))
        return self

    def _date_grid(self) -> list[datetime.datetime]:
        """Every date in ``start``..``end`` at ``frequency``, ignoring ``missing``.

        Used to resolve ``missing`` wildcard patterns; unlike ``values`` it is
        not cached and does not depend on ``missing`` (which is still being
        computed when this runs).
        """
        dates = []
        date = self.start
        while date <= self.end:
            dates.append(date)
            date += self.frequency
        return dates

    @cached_property
    def values(self) -> list[datetime.datetime]:
        missing_set = set(self.missing)
        dates = []
        date = self.start
        while date <= self.end:
            if date not in missing_set:
                dates.append(date)
            date += self.frequency
        return dates

    def start_range(self, dates) -> datetime.datetime:
        """Used for tabular datasets grouping."""
        return dates[0]

    def end_range(self, dates) -> datetime.datetime:
        """Used for tabular datasets grouping."""
        return dates[-1] + self.frequency

    def dump(self, dumper):
        return dumper.start_end_dates(self.start, self.end, self.frequency)


class BaseDates(StartEndDates):
    """Basetimes (forecast initialisation times) for the ``trajectories`` layout.

    Mirrors :class:`StartEndDates`: it models a flat list of base dates via
    ``start`` / ``end`` / ``frequency`` (plus optional ``missing``).  It is kept
    as a distinct type so the ``base_dates`` recipe field is self-documenting and
    can be exported to the JSON schema independently of analysis ``dates``.

    Unlike :class:`StartEndDates`, ``values`` retains the slots for ``missing``
    base dates: the on-disk trajectory array must keep an entry for every base
    date in the range, and :class:`~anemoi.datasets.dates.groups.TrajectoryGroups`
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
    frequency: Frequency = frequency_to_timedelta("1h")
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
