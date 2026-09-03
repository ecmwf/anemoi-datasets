# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import pytest

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate.covering import ForecastCovering
from anemoi.datasets.create.sources.accumulate.covering import ValidTimeCovering


def _hours(n):
    return datetime.timedelta(hours=n)


def _minutes(n):
    return datetime.timedelta(minutes=n)


def test_valid_time_single_field_when_period_equals_length():
    """A base-less 5h source serving a 5h period is one field — 5h need not divide 24h."""
    d = datetime.datetime(2024, 6, 1, 7)  # 07:00, not on any midnight grid
    cover = ValidTimeCovering(length=_hours(5)).cover(d - _hours(5), d)
    assert cover == [SignedInterval(start=d - _hours(5), end=d, base=None)]


def test_valid_time_tiles_and_sums_coarser_period():
    """A 10h period from a 5h base-less source is two summed 5h fields."""
    d = datetime.datetime(2024, 6, 1, 7)
    cover = ValidTimeCovering(length=_hours(5)).cover(d - _hours(10), d)
    assert cover == [
        SignedInterval(start=d - _hours(10), end=d - _hours(5), base=None),
        SignedInterval(start=d - _hours(5), end=d, base=None),
    ]


def test_valid_time_rejects_window_not_multiple_of_length():
    d = datetime.datetime(2024, 6, 1, 7)
    with pytest.raises(ValueError, match="whole multiple of the source increment"):
        ValidTimeCovering(length=_hours(5)).cover(d - _hours(6), d)


def test_valid_time_rejects_imposed_basetime():
    d = datetime.datetime(2024, 6, 1, 7)
    with pytest.raises(NotImplementedError, match="base-less"):
        ValidTimeCovering(length=_hours(5)).cover(d - _hours(5), d, basetime=d - _hours(5))


def test_from_zero_two_intervals():
    """Window [bt+6, bt+12] with from-zero is +a(0,12) - a(0,6)."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero")
    cover = sel.cover(bt + _hours(6), bt + _hours(12), basetime=bt)
    assert len(cover) == 2
    assert cover[0] == SignedInterval(start=bt, end=bt + _hours(12), base=bt)
    assert cover[0].sign == 1
    # negated interval is start>end with same base
    assert cover[1].start == bt + _hours(6)
    assert cover[1].end == bt
    assert cover[1].base == bt
    assert cover[1].sign == -1


def test_from_zero_collapses_when_window_starts_at_basetime():
    """Window [bt, bt+6] with from-zero is the single +a(0,6) interval."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero")
    cover = sel.cover(bt, bt + _hours(6), basetime=bt)
    assert len(cover) == 1
    assert cover[0] == SignedInterval(start=bt, end=bt + _hours(6), base=bt)


def test_increment_single_interval():
    """Window [bt+6, bt+12] with a matching per-step duration is the single a(6,12) interval."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="6h")
    cover = sel.cover(bt + _hours(6), bt + _hours(12), basetime=bt)
    assert cover == [SignedInterval(start=bt + _hours(6), end=bt + _hours(12), base=bt)]


def test_increment_reaccumulates_coarser_period():
    """A period coarser than the increment sums increments: 6h window from 3h fields."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="3h")
    cover = sel.cover(bt + _hours(6), bt + _hours(12), basetime=bt)
    assert cover == [
        SignedInterval(start=bt + _hours(6), end=bt + _hours(9), base=bt),
        SignedInterval(start=bt + _hours(9), end=bt + _hours(12), base=bt),
    ]
    assert all(i.sign == 1 for i in cover)


def test_increment_rejects_window_not_multiple_of_increment():
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(7), accumulation="3h")
    with pytest.raises(ValueError, match="whole multiple of the source increment"):
        sel.cover(bt + _hours(5), bt + _hours(12), basetime=bt)


def test_straddling_basetime_is_rejected():
    bt = datetime.datetime(2021, 1, 1, 12)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero")
    with pytest.raises(ValueError, match="straddles basetime"):
        sel.cover(bt - _hours(3), bt + _hours(3), basetime=bt)


def test_missing_basetime_is_rejected():
    bt = datetime.datetime(2021, 1, 1)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero")
    with pytest.raises(ValueError, match="requires an explicit basetime"):
        sel.cover(bt, bt + _hours(6))


def test_invalid_accumulation_flag_is_rejected():
    with pytest.raises(ValueError, match="Invalid 'accumulation' value"):
        ForecastCovering(period=_hours(6), accumulation="auto")


def test_reset_within_one_cycle():
    """Window [bt+6, bt+12] with 24h reset stays in the first cycle: +a(0,12) - a(0,6)."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero-reset-every-24h")
    cover = sel.cover(bt + _hours(6), bt + _hours(12), basetime=bt)
    assert cover == [
        -SignedInterval(start=bt, end=bt + _hours(6), base=bt),
        SignedInterval(start=bt, end=bt + _hours(12), base=bt),
    ]


def test_reset_second_cycle():
    """Window [bt+30, bt+36] lives in the second cycle: +a(24,36) - a(24,30)."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero-reset-every-24h")
    cover = sel.cover(bt + _hours(30), bt + _hours(36), basetime=bt)
    assert cover == [
        -SignedInterval(start=bt + _hours(24), end=bt + _hours(30), base=bt),
        SignedInterval(start=bt + _hours(24), end=bt + _hours(36), base=bt),
    ]


def test_reset_straddling_boundary():
    """Window [bt+20, bt+26] straddles the 24h reset: -a(0,20) +a(0,24) +a(24,26)."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero-reset-every-24h")
    cover = sel.cover(bt + _hours(20), bt + _hours(26), basetime=bt)
    assert cover == [
        -SignedInterval(start=bt, end=bt + _hours(20), base=bt),
        SignedInterval(start=bt, end=bt + _hours(24), base=bt),
        SignedInterval(start=bt + _hours(24), end=bt + _hours(26), base=bt),
    ]
    assert sum(i.length for i in cover) == _hours(6).total_seconds()


def test_reset_window_starting_on_boundary():
    """Window [bt+24, bt+30]: single interval +a(24,30), no subtraction."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero-reset-every-24h")
    cover = sel.cover(bt + _hours(24), bt + _hours(30), basetime=bt)
    assert cover == [SignedInterval(start=bt + _hours(24), end=bt + _hours(30), base=bt)]


def test_sub_hourly_offsets_are_supported():
    """A window whose endpoints are sub-hourly offsets of the basetime is covered.

    Was rejected outright ("integer-hour offsets"); the decomposition is the
    same signed difference, in whole minutes rather than whole hours.
    """
    bt = datetime.datetime(2021, 1, 1)
    sel = ForecastCovering(period=_minutes(30), accumulation="from-zero")
    cover = sel.cover(bt + _minutes(30), bt + _minutes(60), basetime=bt)
    assert cover == [
        SignedInterval(start=bt, end=bt + _minutes(60), base=bt),
        -SignedInterval(start=bt, end=bt + _minutes(30), base=bt),
    ]


def test_sub_hourly_increment_tiles_the_window():
    """A 30 min window from a 10 min increment source is three summed fields."""
    bt = datetime.datetime(2021, 1, 1)
    sel = ForecastCovering(period=_minutes(30), accumulation="10m")
    cover = sel.cover(bt + _minutes(30), bt + _minutes(60), basetime=bt)
    assert cover == [
        SignedInterval(start=bt + _minutes(30), end=bt + _minutes(40), base=bt),
        SignedInterval(start=bt + _minutes(40), end=bt + _minutes(50), base=bt),
        SignedInterval(start=bt + _minutes(50), end=bt + _minutes(60), base=bt),
    ]


def test_sub_hourly_reset_boundary():
    """Reset every 30 min: a window ending on the boundary is one interval."""
    bt = datetime.datetime(2021, 1, 1)
    sel = ForecastCovering(period=_minutes(20), accumulation="from-zero-reset-every-30m")
    cover = sel.cover(bt + _minutes(10), bt + _minutes(30), basetime=bt)
    assert cover == [
        -SignedInterval(start=bt, end=bt + _minutes(10), base=bt),
        SignedInterval(start=bt, end=bt + _minutes(30), base=bt),
    ]


def test_window_not_multiple_of_increment_rejected():
    """The increment must tile the window exactly, sub-hourly included."""
    bt = datetime.datetime(2021, 1, 1)
    sel = ForecastCovering(period=_minutes(25), accumulation="10m")
    with pytest.raises(ValueError, match="whole\\s+multiple of the source increment"):
        sel.cover(bt + _minutes(5), bt + _minutes(30), basetime=bt)
