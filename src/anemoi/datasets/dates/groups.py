# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import itertools
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterator
from functools import cached_property
from typing import Any

from anemoi.datasets.dates import DatesProvider
from anemoi.datasets.dates import TrajectoryDates
from anemoi.datasets.dates import as_datetime


def _shorten(dates: list[datetime.datetime] | tuple[datetime.datetime, ...]) -> str | list[str]:
    """Shorten the list of dates for display.

    Args:
        dates (Union[List[datetime.datetime], Tuple[datetime.datetime, ...]]): The list of dates.

    Returns:
        Union[str, List[str]]: The shortened list of dates.
    """
    if isinstance(dates, (list, tuple)):
        dates = [d.isoformat() for d in dates]
        if len(dates) > 5:
            return f"{dates[0]}...{dates[-1]}"
    return dates


class GroupOfDates:
    """A class to represent a group of dates."""

    def __init__(self, dates: list[datetime.datetime], provider: DatesProvider, partial_ok: bool = False) -> None:
        assert isinstance(provider, DatesProvider), type(provider)
        assert isinstance(dates, list)

        # Trajectory providers yield ``(basetime, step)`` pairs; they are
        # opaque to ``GroupOfDates`` and stored as-is.
        self.dates = [d if isinstance(d, tuple) else as_datetime(d) for d in dates]
        self.provider = provider
        self.partial_ok = partial_ok

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GroupOfDates":
        """Used in pytest"""
        dates = DatesProvider.from_config(config)
        return cls(dates.values, dates)

    def __len__(self) -> int:
        """Return the number of dates in the group.

        Returns:
            int: The number of dates.
        """
        return len(self.dates)

    def __iter__(self) -> Iterator[datetime.datetime]:
        """Return an iterator over the dates in the group.

        Returns:
            Iterator[datetime.datetime]: The iterator over the dates.
        """
        return iter(self.dates)

    def __repr__(self) -> str:
        """Return a string representation of the group of dates.

        Returns:
            str: The string representation.
        """
        return f"GroupOfDates(dates={_shorten(self.dates)})"

    def __eq__(self, other: object) -> bool:
        """Check if two groups of dates are equal.

        Args:
            other (object): The other group of dates.

        Returns:
            bool: True if the groups are equal, False otherwise.
        """
        return isinstance(other, GroupOfDates) and self.dates == other.dates

    @property
    def start_range(self) -> datetime.datetime:
        import numpy as np

        return np.datetime64(self.provider.start_range(self.dates), "ns")

    @property
    def end_range(self) -> datetime.datetime:
        import numpy as np

        return np.datetime64(self.provider.end_range(self.dates), "ns") - np.timedelta64(1, "ns")


class Groups:
    """A collection of groups of dates.

    Examples:
        >>> list(Groups(group_by="daily", start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=12))[0]
        [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 12, 0)]

        >>> list(Groups(group_by="daily", start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=12))[1]
        [datetime.datetime(2023, 1, 2, 0, 0), datetime.datetime(2023, 1, 2, 12, 0)]

        >>> g = Groups(group_by=3, start="2023-01-01 00:00", end="2023-01-05 00:00", frequency=24)
        >>> len(list(g))
        2
        >>> len(list(g)[0])
        3
        >>> len(list(g)[1])
        2
        >>> g = Groups(group_by=3,
        ...            start="2023-01-01 00:00",
        ...            end="2023-01-05 00:00",
        ...            frequency=24,
        ...            missing=["2023-01-02 00:00"])
        >>> len(list(g))
        2
        >>> len(list(g)[0])
        2
        >>> len(list(g)[1])
        2
    """

    def __init__(self, group_by: Any, **kwargs: Any) -> None:
        """Initialize the class with the provided keyword arguments.

        Parameters
        ----------
            **kwargs : Any : Arbitrary keyword arguments. Expected keys include:
                - group_by: Configuration for the Grouper.
                - Other keys for DatesProvider configuration.
        """

        self._dates = DatesProvider.from_config(**kwargs)
        self._grouper = Grouper.from_config(group_by)
        self._filter = Filter(self._dates.missing)

    @property
    def provider(self) -> DatesProvider:
        """Return the dates provider."""
        return self._dates

    def __iter__(self) -> Iterator[GroupOfDates]:
        """Return an iterator over the groups of dates.

        Returns:
            Iterator[GroupOfDates]: The iterator over the groups of dates.
        """
        for go in self._grouper(self._dates):
            dates = self._filter(go.dates)
            if not dates:
                continue
            yield GroupOfDates(dates, go.provider)

    def __len__(self) -> int:
        """Return the number of groups of dates.

        Returns:
            int: The number of groups.
        """
        return self._len

    @cached_property
    def _len(self) -> int:
        """Calculate the number of groups of dates."""
        n = 0
        for go in self._grouper(self._dates):
            dates = self._filter(go.dates)
            if not dates:
                continue
            n += 1
        return n

    def __repr__(self) -> str:
        """Return a string representation of the groups of dates.

        Returns:
            str: The string representation.
        """
        return f"{self.__class__.__name__}(dates={len(self)},{_shorten(self._dates)})"

    def describe(self) -> str:
        """Return a summary description of the dates.

        Returns:
            str: The summary description.
        """
        return self._dates.summary

    def one_date(self) -> GroupOfDates:
        """Return a group containing only one date.

        Returns:
            GroupOfDates: The group containing only one date.
        """
        go = next(iter(self))
        return GroupOfDates([go.dates[0]], go.provider)

    def first_date(self) -> datetime.datetime:
        """Return the first date across all groups.

        Returns:
            datetime.datetime: The first date.
        """
        return min(self._dates.values)

    def last_date(self) -> datetime.datetime:
        """Return the last date across all groups.

        Returns:
            datetime.datetime: The last date.
        """
        return max(self._dates.values)


class Filter:
    """A class to filter out missing dates."""

    def __init__(self, missing: list[datetime.datetime]) -> None:
        self.missing = {as_datetime(m) for m in missing}

    def __call__(self, dates: list[datetime.datetime]) -> list[datetime.datetime]:
        """Filter out missing dates from the list of dates.

        Args:
            dates (List[datetime.datetime]): The list of dates.

        Returns:
            List[datetime.datetime]: The filtered list of dates.
        """
        return [d for d in dates if d not in self.missing]


class TrajectoryFilter(Filter):
    """Filter out ``(basetime, step)`` pairs whose basetime is missing.

    Used by :class:`TrajectoryGroups` because the items yielded by the grouper
    are pairs, not plain datetimes.  The plain :class:`Filter` would compare a
    pair to a set of datetimes and find no match — silently leaving missing
    base dates in the iteration.
    """

    def __call__(self, pairs):
        return [(bt, st) for (bt, st) in pairs if bt not in self.missing]


class Grouper(ABC):
    """Abstract base class for grouping dates."""

    @classmethod
    def from_config(cls, group_by: Any) -> "Grouper":
        """Create a grouper based on the configuration."""

        if isinstance(group_by, int) and group_by > 0:
            return GrouperByFixedSize(group_by)

        if group_by is None:
            return GrouperOneGroup()

        if group_by == "reference_date":
            return ReferenceDateGroup()

        key = {
            "monthly": lambda dt: (dt.year, dt.month),
            "daily": lambda dt: (dt.year, dt.month, dt.day),
            "weekly": lambda dt: (dt.weekday(),),
            "yearly": lambda dt: (dt.year,),
            "MMDD": lambda dt: (dt.month, dt.day),
        }[group_by]
        return GrouperByKey(key)

    @abstractmethod
    def __call__(self, dates: DatesProvider) -> Iterator[GroupOfDates]:
        """Group dates based on the implementation.

        Args:
            dates (DatesProvider): The dates provider.

        Returns:
            Iterator[GroupOfDates]: The iterator over the groups of dates.
        """
        pass


class ReferenceDateGroup(Grouper):
    """Group dates by their reference date."""

    def __call__(self, dates: DatesProvider) -> Iterator[GroupOfDates]:
        """Group dates by their reference date.

        Args:
            dates (DatesProvider): The dates provider.

        Returns:
            Iterator[GroupOfDates]: The iterator over the groups of dates.
        """
        assert isinstance(dates, DatesProvider), type(dates)

        mapping = dates.mapping

        def same_refdate(dt):
            return mapping[dt].refdate

        for _, g in itertools.groupby(sorted(dates, key=same_refdate), key=same_refdate):
            yield GroupOfDates(list(g), dates)


class GrouperOneGroup(Grouper):
    """Group all dates into a single group."""

    def __call__(self, dates: DatesProvider) -> Iterator[GroupOfDates]:
        """Group all dates into a single group.

        Args:
            dates (DatesProvider): The dates provider.

        Returns:
            Iterator[GroupOfDates]: The iterator over the groups of dates.
        """
        assert isinstance(dates, DatesProvider), type(dates)

        yield GroupOfDates(dates.values, dates)


class GrouperByKey(Grouper):
    """Group dates by a key."""

    def __init__(self, key: Callable[[datetime.datetime], Any]) -> None:
        self.key = key

    def _key(self, d: Any) -> Any:
        # Trajectory providers yield ``(basetime, step)`` pairs; apply the key
        # to the basetime so monthly/daily/yearly grouping still makes sense.
        if isinstance(d, tuple):
            return self.key(d[0])
        return self.key(d)

    def __call__(self, dates: DatesProvider) -> Iterator[GroupOfDates]:
        """Group dates based on the provided key.

        Args:
            dates (DatesProvider): The dates provider.

        Returns:
            Iterator[GroupOfDates]: The iterator over the groups of dates.
        """
        for _, g in itertools.groupby(sorted(dates, key=self._key), key=self._key):
            yield GroupOfDates(list(g), dates)


class GrouperByFixedSize(Grouper):
    """Group dates by a fixed size."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, dates: DatesProvider) -> Iterator[GroupOfDates]:
        """Group dates into fixed-size batches.

        Args:
            dates (DatesProvider): The dates provider.

        Returns:
            Iterator[GroupOfDates]: The iterator over the groups of dates.
        """
        batch = []

        for d in dates:
            batch.append(d)
            if len(batch) == self.size:
                yield GroupOfDates(batch, dates)
                batch = []

        if batch:
            yield GroupOfDates(batch, dates)


class TrajectoryGroups(Groups):
    """Groups whose provider is a :class:`TrajectoryDates`.

    Values iterated by the provider are ``(basetime, step)`` pairs rather than
    plain datetimes, so ``first_date`` / ``last_date`` are sourced from
    :meth:`TrajectoryDates.factorise` (basetimes only) to keep metadata
    meaningful.

    Parameters
    ----------
    steps : dict
        ``recipe.steps`` mapping (``start``, ``end``, ``frequency``) describing
        the list of forecast steps.
    group_by : Any
        Grouping configuration forwarded to :meth:`Grouper.from_config`.
    base_dates : Any
        Configuration forwarded to :class:`TrajectoryDates` to build the
        underlying basetimes provider (``start``, ``end``, ``frequency``,
        ``missing``, …).
    **kwargs : Any
        Additional keyword arguments forwarded to :class:`TrajectoryDates`.
    """

    def __init__(self, steps: Any, group_by: Any, base_dates: Any, **kwargs: Any) -> None:
        self._dates = TrajectoryDates(steps=steps, **base_dates, **kwargs)
        self._grouper = Grouper.from_config(group_by)
        self._filter = TrajectoryFilter(self._dates.missing)

    def __iter__(self):
        import numpy as np

        from anemoi.datasets.create.arguments import ForecastDates

        for go in self._grouper(self._dates):
            pairs = self._filter(go.dates)
            if not pairs:
                continue
            items = []
            for basetime, step_td in pairs:
                step_seconds = int(step_td / np.timedelta64(1, "s"))
                step = datetime.timedelta(seconds=step_seconds)
                items.append((basetime + step, basetime))
            yield ForecastDates(items)

    def one_date(self):
        """Return a single-item ForecastDates for minimal input probing."""
        go = next(iter(self))
        from anemoi.datasets.create.arguments import ForecastDates

        return ForecastDates([go.items[0]])

    def first_date(self) -> datetime.datetime:
        basetimes, steps = self._dates.factorise()
        step_start = steps[0].astype("timedelta64[s]").astype(datetime.timedelta)
        return min(basetimes) + step_start

    def last_date(self) -> datetime.datetime:
        basetimes, steps = self._dates.factorise()
        step_end = steps[-1].astype("timedelta64[s]").astype(datetime.timedelta)
        return max(basetimes) + step_end
