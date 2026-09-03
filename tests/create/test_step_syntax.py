# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The archive step syntax, and the invariant that whole-hour steps do not move.

``step_to_timedelta`` / ``timedelta_to_step`` are the single boundary between
the pipeline's internal timedeltas and the ``step`` key of a MARS/FDB request
(or a GRIB message's step keys).  The asymmetry is deliberate: a whole-hour
step stays a plain ``int`` so that a request built from an hour-based recipe is
byte-for-byte what it has always been -- the cached test fixtures are named
after the hash of the request dict, so a change of representation would rename
every one of them.
"""

import datetime
import json

import pytest

from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.intervals import step_to_timedelta
from anemoi.datasets.create.intervals import timedelta_to_step


def _hours(n):
    return datetime.timedelta(hours=n)


def _minutes(n):
    return datetime.timedelta(minutes=n)


@pytest.mark.parametrize(
    "step,expected",
    [
        (0, datetime.timedelta(0)),
        (12, _hours(12)),
        ("12", _hours(12)),
        ("24h", _hours(24)),
        ("10m", _minutes(10)),
        ("60m", _hours(1)),
        ("10h10m", _minutes(610)),
        ("1h30m", _minutes(90)),
        (_minutes(10), _minutes(10)),
    ],
)
def test_step_to_timedelta(step, expected):
    assert step_to_timedelta(step) == expected


@pytest.mark.parametrize("bad", ["", "10s", "abc", "1d", "10m10h", "-1h", 1.5, True])
def test_step_to_timedelta_rejects(bad):
    with pytest.raises(ValueError):
        step_to_timedelta(bad)


@pytest.mark.parametrize(
    "offset,expected",
    [
        (datetime.timedelta(0), 0),
        (_hours(12), 12),
        (_hours(240), 240),
        (_minutes(10), "10m"),
        (_minutes(30), "30m"),
        (_minutes(610), "10h10m"),
        (_minutes(90), "1h30m"),
    ],
)
def test_timedelta_to_step(offset, expected):
    assert timedelta_to_step(offset) == expected


def test_timedelta_to_step_rejects():
    with pytest.raises(ValueError, match="negative"):
        timedelta_to_step(-_hours(1))
    with pytest.raises(ValueError, match="whole number of minutes"):
        timedelta_to_step(datetime.timedelta(seconds=30))


def test_round_trip():
    for minutes in list(range(0, 200)) + [600, 610, 1440, 14400]:
        offset = _minutes(minutes)
        assert step_to_timedelta(timedelta_to_step(offset)) == offset


# ---------------------------------------------------------------------------
# The invariant: whole-hour steps keep their representation
# ---------------------------------------------------------------------------


def test_whole_hours_stay_plain_ints():
    """A whole-hour step is an ``int``, never a string.

    ``tests/create/utils/mock_sources.py`` names its cached GRIB fixture after
    ``md5(json.dumps([args, kwargs], sort_keys=True, default=str))``, so
    ``{"step": 6}`` and ``{"step": "6h"}`` are different files.  Returning a
    string here would 404 every uploaded fixture.
    """
    for n in range(0, 385):
        step = timedelta_to_step(_hours(n))
        assert isinstance(step, int) and not isinstance(step, bool), (n, step)
        assert step == n


def test_request_hash_of_an_hourly_request_is_unchanged():
    """A frozen hash of an hour-based request, as the mock source computes it."""
    request = {"class": "od", "param": "tp", "date": "20200101", "time": "0000", "step": 6}
    string = json.dumps([(), request], sort_keys=True, default=str)
    assert '"step": 6' in string  # not "6", not "6h", not 6.0

    import hashlib

    assert hashlib.md5(string.encode("utf8")).hexdigest() == "f4642b5efabf7faa5e43b25a66954328"


def test_adjust_request_encodes_the_step():
    """``Intervals.adjust_request`` is the accumulate -> archive boundary."""
    base = datetime.datetime(2020, 1, 1)
    hourly = SignedInterval(start=base, end=base + _hours(6), base=base)
    sub_hourly = SignedInterval(start=base, end=base + _minutes(10), base=base)

    argument = Intervals(dates=[base + _hours(6)], intervals=[hourly, sub_hourly])

    _, request, step = argument.adjust_request(hourly, {"param": "tp"})
    assert step == 6 and request["step"] == 6
    assert request["date"] == "20200101" and request["time"] == "0000"

    _, request, step = argument.adjust_request(sub_hourly, {"param": "tp"})
    assert step == "10m" and request["step"] == "10m"


def test_forecast_adjust_request_encodes_the_step():
    """Same boundary on the trajectory side."""
    base = datetime.datetime(2020, 1, 1)
    interval = SignedInterval(start=base + _minutes(20), end=base + _minutes(30), base=base)
    argument = ForecastIntervals(items=[(base + _minutes(30), base, _minutes(10))], intervals=[interval])

    _, request, step = argument.adjust_request(interval, {"param": "tp"})
    assert step == "30m" and request["step"] == "30m"
