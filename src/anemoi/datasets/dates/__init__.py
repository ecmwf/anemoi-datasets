# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import warnings
from functools import reduce
from math import gcd
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import DateTimes
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.hindcasts import HindcastDatesTimes
from anemoi.utils.humanize import print_dates


def extend(x: Union[str, List[Any], Tuple[Any, ...]]) -> Iterator[datetime.datetime]:
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


class DatesProvider:
    """Base class for date generation.

    >>> DatesProvider.from_config(**{"start": "2023-01-01 00:00", "end": "2023-01-02 00:00", "frequency": "1d"}).values
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 2, 0, 0)]

    >>> DatesProvider.from_config(**{"start": "2023-01-01 00:00", "end": "2023-01-03 00:00", "frequency": "18h"}).values
    [datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 1, 1, 18, 0), datetime.datetime(2023, 1, 2, 12, 0)]

    >>> DatesProvider.from_config(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=6).as_dict()
    {'start': '2023-01-01T00:00:00', 'end': '2023-01-02T00:00:00', 'frequency': '6h'}

    >>> len(DatesProvider.from_config(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=12))
    3
    >>> len(DatesProvider.from_config(start="2023-01-01 00:00",
    ...                   end="2023-01-02 00:00",
    ...                   frequency=12,
    ...                   missing=["2023-01-01 12:00"]))
    3
    >>> len(DatesProvider.from_config(start="2023-01-01 00:00",
    ...                   end="2023-01-02 00:00",
    ...                   frequency=12,
    ...                   missing=["2099-01-01 12:00"]))
    3
    """

    def __init__(self, missing: Optional[List[Union[str, datetime.datetime]]] = None) -> None:
        """Initialize the DatesProvider with optional missing dates.

        Parameters
        ----------
        missing : Optional[List[Union[str, datetime.datetime]]]
            List of missing dates.
        """
        if not missing:
            missing = []
        self.missing = list(extend(missing))
        if set(self.missing) - set(self.values):
            diff = set(self.missing) - set(self.values)
            warnings.warn(f"Missing dates {len(diff)=} not in list.")

    @classmethod
    def from_config(cls, **kwargs: Any) -> "DatesProvider":
        """Create a DatesProvider instance from configuration.

        Args:
            **kwargs (Any): Configuration parameters.

        Returns
        -------
        DatesProvider
            An instance of DatesProvider.
        """
        if kwargs.pop("hindcasts", False):
            return HindcastsDates(**kwargs)

        if "values" in kwargs:
            return ValuesDates(**kwargs)

        return StartEndDates(**kwargs)

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

    @property
    def summary(self) -> str:
        """Get a summary of the date range."""
        return f"📅 {self.values[0]} ... {self.values[-1]}"


class ValuesDates(DatesProvider):
    """Class for handling a list of date values.

    Args:
        values (List[Union[str, datetime.datetime]]): List of date values.
        **kwargs (Any): Additional arguments.
    """

    def __init__(self, values: List[Union[str, datetime.datetime]], **kwargs: Any) -> None:
        """Initialize ValuesDates with a list of values.

        Args:
            values (List[Union[str, datetime.datetime]]): List of date values.
            **kwargs (Any): Additional arguments.
        """
        self.values = sorted([as_datetime(_) for _ in values])
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Get a string representation of the ValuesDates instance.

        Returns
        -------
        str
            String representation of the instance.
        """
        return f"{self.__class__.__name__}({self.values[0]}..{self.values[-1]})"

    def as_dict(self) -> Dict[str, Any]:
        """Convert the ValuesDates instance to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the instance.
        """
        return {"values": self.values[0]}


class StartEndDates(DatesProvider):
    """Class for generating dates between a start and end date with a specified frequency.

    Args:
        start (Union[str, datetime.datetime]): Start date.
        end (Union[str, datetime.datetime]): End date.
        frequency (Union[int, str]): Frequency of dates.
        **kwargs (Any): Additional arguments.
    """

    def __repr__(self) -> str:
        """Get a string representation of the StartEndDates instance."""
        return f"{self.__class__.__name__}({self.start}..{self.end} every {self.frequency})"

    def __init__(
        self,
        start: Union[str, datetime.datetime],
        end: Union[str, datetime.datetime],
        frequency: Union[int, str] = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize StartEndDates with start, end, and frequency.

        Args:
            start (Union[str, datetime.datetime]): Start date.
            end (Union[str, datetime.datetime]): End date.
            frequency (Union[int, str]): Frequency of dates.
            **kwargs (Any): Additional arguments.
        """
        frequency = frequency_to_timedelta(frequency)
        assert isinstance(frequency, datetime.timedelta), frequency

        def _(x):
            if isinstance(x, str):
                return datetime.datetime.fromisoformat(x)
            return x

        start = _(start)
        end = _(end)

        if isinstance(start, datetime.date) and not isinstance(start, datetime.datetime):
            start = datetime.datetime(start.year, start.month, start.day)

        if isinstance(end, datetime.date) and not isinstance(end, datetime.datetime):
            end = datetime.datetime(end.year, end.month, end.day)

        start = as_datetime(start)
        end = as_datetime(end)

        self.start = start
        self.end = end
        self.frequency = frequency

        missing = kwargs.pop("missing", [])

        self.values = list(DateTimes(start, end, increment=frequency, **kwargs))
        self.kwargs = kwargs

        super().__init__(missing=missing)

    def as_dict(self) -> Dict[str, Any]:
        """Convert the StartEndDates instance to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the instance.
        """
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "frequency": frequency_to_string(self.frequency),
        }.update(self.kwargs)


class Hindcast:
    """Class representing a single hindcast date.

    Args:
        date (datetime.datetime): The date of the hindcast.
        refdate (datetime.datetime): The reference date.
        hdate (datetime.datetime): The hindcast date.
        step (int): The step value.
    """

    def __init__(
        self, date: datetime.datetime, refdate: datetime.datetime, hdate: datetime.datetime, step: int
    ) -> None:
        """Initialize a Hindcast instance.

        Args:
            date (datetime.datetime): The date of the hindcast.
            refdate (datetime.datetime): The reference date.
            hdate (datetime.datetime): The hindcast date.
            step (int): The step value.
        """
        self.date = date
        self.refdate = refdate
        self.hdate = hdate
        self.step = step


class HindcastsDates(DatesProvider):
    """Class for generating hindcast dates over a range of years.

    Args:
        start (Union[str, List[str]]): Start date(s).
        end (Union[str, List[str]]): End date(s).
        steps (List[int]): List of step values.
        years (int): Number of years.
        **kwargs (Any): Additional arguments.
    """

    def __init__(
        self,
        start: Union[str, List[str]],
        end: Union[str, List[str]],
        steps: List[int] = [0],
        years: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initialize HindcastsDates with start, end, steps, and years.

        Args:
            start (Union[str, List[str]]): Start date(s).
            end (Union[str, List[str]]): End date(s).
            steps (List[int]): List of step values.
            years (int): Number of years.
            **kwargs (Any): Additional arguments.
        """
        if not isinstance(start, list):
            start = [start]
            end = [end]

        reference_dates = []
        for s, e in zip(start, end):
            reference_dates.extend(list(DateTimes(s, e, increment=24, **kwargs)))
        # reference_dates = list(DateTimes(start, end, increment=24, **kwargs))
        dates = []

        seen = {}

        for hdate, refdate in HindcastDatesTimes(reference_dates=reference_dates, years=years):
            assert refdate - hdate >= datetime.timedelta(days=365), (refdate - hdate, refdate, hdate)
            for step in steps:

                date = hdate + datetime.timedelta(hours=step)

                if date in seen:
                    raise ValueError(f"Duplicate date {date}={hdate}+{step} for {refdate} and {seen[date]}")

                seen[date] = Hindcast(date, refdate, hdate, step)

                assert refdate - date > datetime.timedelta(days=360), (refdate - date, refdate, date, hdate, step)

                dates.append(date)

        dates = sorted(dates)

        deltas = set()
        for a, b in zip(dates, dates[1:]):
            delta = b - a
            assert isinstance(delta, datetime.timedelta), delta
            deltas.add(delta)

        mindelta_seconds = reduce(gcd, [int(delta.total_seconds()) for delta in deltas])
        mindelta = datetime.timedelta(seconds=mindelta_seconds)
        self.frequency = mindelta
        assert mindelta.total_seconds() > 0, mindelta
        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥", dates[0], dates[-1], mindelta)

        # Use all values between start and end by frequency, and set the ones that are missing
        self.values = []
        missing = []
        date = dates[0]
        last = date
        print("------", date, dates[-1])
        dateset = set(dates)
        while date <= dates[-1]:
            self.values.append(date)
            if date not in dateset:
                missing.append(date)
                seen[date] = seen[last]
            else:
                last = date
            date = date + mindelta

        self.mapping = seen

        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥", self.values[0], self.values[-1], mindelta)
        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥", f"{len(self.values)=} - {len(missing)=}")

        super().__init__(missing=missing)

    def __repr__(self) -> str:
        """Get a string representation of the HindcastsDates instance.

        Returns
        -------
        str
            String representation of the instance.
        """
        return f"{self.__class__.__name__}({self.values[0]}..{self.values[-1]})"

    def as_dict(self) -> Dict[str, Any]:
        """Convert the HindcastsDates instance to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the instance.
        """
        return {"hindcasts": self.hindcasts}


if __name__ == "__main__":
    print_dates([datetime.datetime(2023, 1, 1, 0, 0)])
    s = StartEndDates(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=1)
    print_dates(list(s))
