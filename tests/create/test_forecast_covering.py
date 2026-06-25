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


def _hours(n):
    return datetime.timedelta(hours=n)


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


def test_from_previous_step_single_interval():
    """Window [bt+6, bt+12] with from-previous-step is the single a(6,12) interval."""
    bt = datetime.datetime(2021, 1, 1, 0)
    sel = ForecastCovering(period=_hours(6), accumulation="from-previous-step")
    cover = sel.cover(bt + _hours(6), bt + _hours(12), basetime=bt)
    assert cover == [SignedInterval(start=bt + _hours(6), end=bt + _hours(12), base=bt)]


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
    with pytest.raises(ValueError, match="Invalid accumulation"):
        ForecastCovering(period=_hours(6), accumulation="auto")


def test_non_integer_hours_rejected():
    bt = datetime.datetime(2021, 1, 1)
    sel = ForecastCovering(period=_hours(6), accumulation="from-zero")
    with pytest.raises(ValueError, match="integer-hour"):
        sel.cover(bt + datetime.timedelta(minutes=30), bt + _hours(6), basetime=bt)
