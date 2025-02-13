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

# from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import DateTimes
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import datetimes_factory
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.hindcasts import HindcastDatesTimes
from anemoi.utils.humanize import print_dates


def extend(x):

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

    def __init__(self, missing=None):
        if not missing:
            missing = []
        self.missing = list(extend(missing))
        if set(self.missing) - set(self.values):
            diff = set(self.missing) - set(self.values)
            warnings.warn(f"Missing dates {len(diff)=} not in list.")

    @classmethod
    def from_config(cls, **kwargs):

        if kwargs.pop("fake_forecasts", False):
            return FakeForecastsDates(**kwargs)

        if kwargs.pop("fake_hindcasts", False):
            return FakeHindcastsDates(**kwargs)

        if kwargs.pop("hindcasts", False):
            return HindcastsDates(**kwargs)

        if "values" in kwargs:
            return ValuesDates(**kwargs)

        return StartEndDates(**kwargs)

    def __iter__(self):
        yield from self.values

    def __getitem__(self, i):
        return self.values[i]

    def __len__(self):
        return len(self.values)

    @property
    def summary(self):
        return f"📅 {self.values[0]} ... {self.values[-1]}"

    def patch_result(self, result):
        # Give an opportunity to patch the result (e.g. change the valid_time)
        return result

    def check_fake_support_function(self, function, fake_function):
        return function

    def metadata(self):
        return {}


class ValuesDates(DatesProvider):
    def __init__(self, values, **kwargs):
        self.values = sorted([as_datetime(_) for _ in values])
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values[0]}..{self.values[-1]})"

    def as_dict(self):
        return {"values": self.values[0]}


class StartEndDates(DatesProvider):
    def __init__(self, start, end, frequency=1, **kwargs):

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

    def as_dict(self):
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "frequency": frequency_to_string(self.frequency),
        }.update(self.kwargs)


class Forecast:

    def __init__(self, fakedate, date, step):
        self.fakedate = fakedate
        self.date = date
        self.step = step
        self.valid_datetime = date + datetime.timedelta(hours=step)
        self.metadata = dict(
            date=int(self.date.strftime("%Y%m%d")), step=self.step, time=int(self.date.strftime("%H%M"))
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metadata})"


class Hindcast:

    def __init__(self, fakedate, refdate, hdate, step):
        self.fakedate = fakedate
        self.refdate = refdate
        self.hdate = hdate
        self.step = step
        self.valid_datetime = hdate + datetime.timedelta(hours=step)
        self.metadata = dict(
            hdate=int(self.hdate.strftime("%Y%m%d")),
            date=int(self.refdate.strftime("%Y%m%d")),
            step=self.step,
            time=0,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.metadata})"


class HindcastsDates(DatesProvider):
    def __init__(self, start, end, steps=[0], years=20, **kwargs):

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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values[0]}..{self.values[-1]})"

    def as_dict(self):
        return {"hindcasts": self.hindcasts}


class FakeDateProvider(DatesProvider):

    def check_fake_support(self, action):
        assert action.supports_fake_dates, action

    def check_fake_support_function(self, function, fake_function):
        assert fake_function, function
        if callable(fake_function):
            return fake_function
        return function


class FakeHindcastsDates(FakeDateProvider):
    def __init__(self, start, end, steps, years=20, **kwargs):
        dates = datetimes_factory(
            name="hindcast", reference_dates=dict(start=start, end=end, day_of_week=["monday", "thursday"]), years=years
        )
        all_dates = {}
        ref = datetime.datetime(1900, 1, 1, 0, 0)
        for hdate, refdate in dates:
            for s in steps:
                all_dates[(refdate, hdate, s)] = ref + datetime.timedelta(hours=len(all_dates))

        self.frequency = datetime.timedelta(hours=1)

        self.values = sorted(all_dates.values())
        self.mapping = {v: Hindcast(v, *k) for k, v in all_dates.items()}
        self.date_mapping = all_dates

        super().__init__(missing=[])

    def patch_result(self, result):
        from anemoi.transform.fields import new_field_with_valid_datetime
        from anemoi.transform.fields import new_fieldlist_from_list
        from earthkit.data.utils.dates import to_datetime

        from ..create.input.result import Result

        ds = result.datasource
        data = []
        for field in ds:
            # We get them one at a time because wrappers do not support lists
            refdate, hdate, step, time = (
                field.metadata("date"),
                field.metadata("hdate"),
                field.metadata("step"),
                field.metadata("time"),
            )
            assert time == 0, time
            newdate = self.date_mapping[(to_datetime(refdate), to_datetime(hdate), step)]
            data.append(new_field_with_valid_datetime(field, newdate))

        class PatchedResult(Result):
            def __init__(self, data):
                super().__init__(result.context, result.action_path, result.group_of_dates)
                self._datasource = new_fieldlist_from_list(data)

            @property
            def datasource(self):
                return self._datasource

        return PatchedResult(data)

    def metadata(self):
        return {"fake_hindcasts": {v.isoformat(): k for k, v in self.date_mapping.items()}}


class FakeForecastsDates(FakeDateProvider):
    def __init__(self, start, end, steps, frequency=24, **kwargs):
        dates = datetimes_factory(
            start=start,
            end=end,
            frequency=frequency,
        )
        all_dates = {}
        ref = datetime.datetime(1900, 1, 1, 0, 0)
        for date in dates:
            for s in steps:
                all_dates[(date, s)] = ref + datetime.timedelta(hours=len(all_dates))

        self.frequency = datetime.timedelta(hours=1)

        self.values = sorted(all_dates.values())
        self.mapping = {v: Forecast(v, *k) for k, v in all_dates.items()}
        self.date_mapping = all_dates

        super().__init__(missing=[])

    def actual_dates(self, dates):
        return [self.date_mapping[d] for d in dates]

    def patch_result(self, result):
        from anemoi.transform.fields import new_field_with_valid_datetime
        from anemoi.transform.fields import new_fieldlist_from_list
        from earthkit.data.utils.dates import to_datetime

        from ..create.input.result import Result

        ds = result.datasource
        data = []
        for field in ds:
            # We get them one at a time because wrappers do not support lists
            date, step, time = (
                field.metadata("date"),
                field.metadata("step"),
                field.metadata("time"),
            )
            assert time == 0, time
            newdate = self.date_mapping[(to_datetime(date), step)]
            data.append(new_field_with_valid_datetime(field, newdate))

        class PatchedResult(Result):
            def __init__(self, data):
                super().__init__(result.context, result.action_path, result.group_of_dates)
                self._datasource = new_fieldlist_from_list(data)

            @property
            def datasource(self):
                return self._datasource

        return PatchedResult(data)

    def metadata(self):
        return {"fake_forecasts": {v.isoformat(): k for k, v in self.date_mapping.items()}}


if __name__ == "__main__":
    print_dates([datetime.datetime(2023, 1, 1, 0, 0)])
    s = StartEndDates(start="2023-01-01 00:00", end="2023-01-02 00:00", frequency=1)
    print_dates(list(s))
