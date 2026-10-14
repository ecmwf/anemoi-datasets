# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A covering partitions a window into subwindows, each rebuilt from archived fields."""

import datetime

import pytest

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.windowed.covering import ForecastCovering
from anemoi.datasets.create.sources.windowed.covering import ValidTimeCovering
from anemoi.datasets.create.sources.windowed.subwindows import Subwindow
from anemoi.datasets.create.sources.windowed.subwindows import contributions_of
from anemoi.datasets.create.sources.windowed.subwindows import direct
from anemoi.datasets.create.sources.windowed.subwindows import group_intervals_into_subwindows
from anemoi.datasets.create.sources.windowed.subwindows import validate_contributions
from anemoi.datasets.create.sources.windowed.subwindows import validate_partition

BASE = datetime.datetime(2021, 1, 1, 0)


def _hours(n):
    return datetime.timedelta(hours=n)


def _interval(start_h, end_h, base=BASE):
    return SignedInterval(start=BASE + _hours(start_h), end=BASE + _hours(end_h), base=base)


# ── the subwindow itself ─────────────────────────────────────────────


def test_a_direct_subwindow_is_its_own_contribution():
    subwindow = direct(_interval(0, 1))
    assert subwindow.is_direct
    assert subwindow.contributions == (_interval(0, 1),)


def test_a_differenced_subwindow_is_not_direct():
    subwindow = Subwindow(interval=_interval(6, 12), contributions=(_interval(0, 12), -_interval(0, 6)))
    assert not subwindow.is_direct


def test_contributions_flatten_in_order():
    subwindows = [direct(_interval(0, 1)), direct(_interval(1, 2))]
    assert contributions_of(subwindows) == [_interval(0, 1), _interval(1, 2)]


# ── the window-level invariant ───────────────────────────────────────


def test_partition_accepts_a_contiguous_cover():
    validate_partition([direct(_interval(i, i + 1)) for i in range(6)], BASE, BASE + _hours(6))


def test_partition_rejects_a_gap():
    with pytest.raises(ValueError, match="gap"):
        validate_partition([direct(_interval(0, 2)), direct(_interval(3, 6))], BASE, BASE + _hours(6))


def test_partition_rejects_an_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_partition([direct(_interval(0, 4)), direct(_interval(3, 6))], BASE, BASE + _hours(6))


def test_partition_rejects_overhang():
    with pytest.raises(ValueError, match="does not line up"):
        validate_partition([direct(_interval(0, 9))], BASE, BASE + _hours(6))


def test_partition_rejects_an_empty_covering():
    with pytest.raises(ValueError, match="Empty covering"):
        validate_partition([], BASE, BASE + _hours(6))


# ── the subwindow-level invariant ────────────────────────────────────


def test_contributions_may_be_a_difference():
    validate_contributions(Subwindow(interval=_interval(6, 12), contributions=(_interval(0, 12), -_interval(0, 6))))


def test_contributions_that_do_not_add_up_are_rejected():
    subwindow = Subwindow(interval=_interval(6, 12), contributions=(_interval(0, 9), -_interval(0, 6)))
    with pytest.raises(ValueError, match="do not reconstruct it"):
        validate_contributions(subwindow)


# ── grouping a chain ─────────────────────────────────────────────────


def test_a_forward_chain_becomes_one_subwindow_per_link():
    """A per-step archive: every link advances the partition."""
    chain = [_interval(i, i + 1) for i in range(6)]
    subwindows = group_intervals_into_subwindows(chain, BASE, BASE + _hours(6))
    assert len(subwindows) == 6
    assert all(s.is_direct for s in subwindows)


def test_a_chain_dipping_outside_the_window_stays_one_subwindow():
    """A from-zero archive reaches [6,12] via the basetime, which is outside the window.

    Those two links are one reconstructed part, so the walk must not close a
    subwindow at the basetime.
    """
    chain = [-_interval(0, 6), _interval(0, 12)]
    subwindows = group_intervals_into_subwindows(chain, BASE + _hours(6), BASE + _hours(12))

    assert len(subwindows) == 1
    assert not subwindows[0].is_direct
    assert subwindows[0].start == BASE + _hours(6)
    assert subwindows[0].end == BASE + _hours(12)
    assert len(subwindows[0].contributions) == 2


def test_a_chain_that_overshoots_leaves_the_walk_mid_reconstruction():
    with pytest.raises(ValueError, match="ended mid-reconstruction"):
        group_intervals_into_subwindows([_interval(0, 9)], BASE, BASE + _hours(6))


def test_a_chain_that_stops_short_fails_the_partition():
    with pytest.raises(ValueError, match="does not line up"):
        group_intervals_into_subwindows([_interval(0, 3)], BASE, BASE + _hours(6))


# ── through the real coverings ───────────────────────────────────────


def test_from_zero_is_one_differenced_subwindow():
    covering = ForecastCovering(period=_hours(6), accumulation="from-zero")
    subwindows = covering.partition(BASE + _hours(6), BASE + _hours(12), basetime=BASE)

    assert len(subwindows) == 1
    assert not subwindows[0].is_direct
    # and it still asks the archive for exactly the two fields, in the same order
    assert contributions_of(subwindows) == [_interval(0, 12), -_interval(0, 6)]


def test_from_zero_at_the_basetime_is_direct():
    """The window starts at the basetime, so there is nothing to subtract."""
    covering = ForecastCovering(period=_hours(6), accumulation="from-zero")
    subwindows = covering.partition(BASE, BASE + _hours(6), basetime=BASE)
    assert len(subwindows) == 1 and subwindows[0].is_direct


def test_per_step_increments_are_direct_subwindows():
    covering = ForecastCovering(period=_hours(6), accumulation="3h")
    subwindows = covering.partition(BASE + _hours(6), BASE + _hours(12), basetime=BASE)
    assert len(subwindows) == 2
    assert all(s.is_direct for s in subwindows)


def test_valid_time_source_is_direct_subwindows():
    covering = ValidTimeCovering(length=_hours(5))
    subwindows = covering.partition(BASE, BASE + _hours(10))
    assert len(subwindows) == 2
    assert all(s.is_direct for s in subwindows)
    validate_partition(subwindows, BASE, BASE + _hours(10))
