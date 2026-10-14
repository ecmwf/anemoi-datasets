# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The interval-valued loop: cutting windows, and rebuilding and reducing subwindows."""

import datetime

import numpy as np
import pytest

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.windowed.check import Pieces
from anemoi.datasets.create.sources.windowed.check import check
from anemoi.datasets.create.sources.windowed.common import check_complete
from anemoi.datasets.create.sources.windowed.covering import covering_from_description
from anemoi.datasets.create.sources.windowed.description import validate_from
from anemoi.datasets.create.sources.windowed.operations import OPERATIONS
from anemoi.datasets.create.sources.windowed.ranges import argument
from anemoi.datasets.create.sources.windowed.ranges import cut
from anemoi.datasets.create.sources.windowed.reducer import Reducer
from anemoi.datasets.create.sources.windowed.subwindows import Subwindow
from anemoi.datasets.create.sources.windowed.subwindows import contributions_of

H = datetime.timedelta(hours=1)
PERIOD = 6 * H

#: A from-zero forecast archive: every field is the total since the 00Z run started.
CUMULATIVE = {
    "base_dates": {"times": [0]},
    "steps": {"start": "1h", "end": "24h", "frequency": "1h"},
    "accumulation": "from-zero",
}

#: The wind-gust layout: hourly to step 3, then 3-hourly.
IRREGULAR = {"base_dates": {"times": [0]}, "steps": ["0-1", "1-2", "2-3", "3-6", "6-9", "9-12"]}

NOON = datetime.datetime(2021, 1, 1, 12)
SIX = datetime.datetime(2021, 1, 1, 6)


def _cut(from_: dict, valid_date: datetime.datetime, over=None) -> list:
    covering = covering_from_description(validate_from(from_), period=PERIOD, source_name="test")
    return cut(covering, valid_date - PERIOD, valid_date, over=over)


def _hours(subwindows) -> list:
    return [(s.end - s.start) // H for s in subwindows]


def test_an_archive_of_hourly_fields_is_six_direct_subwindows() -> None:
    subwindows = _cut({"accumulation": "1h"}, SIX)
    assert _hours(subwindows) == [1] * 6
    assert all(s.is_direct for s in subwindows)


def test_an_irregular_archive_is_expressible() -> None:
    """The case ``from: {frequency: ...}`` cannot state."""
    subwindows = _cut(IRREGULAR, datetime.datetime(2021, 1, 1, 9))
    assert _hours(subwindows) == [3, 3]
    assert all(s.is_direct for s in subwindows)


def test_a_cumulative_archive_without_over_is_one_differenced_subwindow() -> None:
    (subwindow,) = _cut(CUMULATIVE, NOON)
    assert not subwindow.is_direct
    # fine for a sum, degenerate for an extremum
    pieces = Pieces.of_subwindows([[subwindow]])
    check("sum", "accum", pieces)
    with pytest.raises(ValueError, match="single piece"):
        check("max", "accum", pieces)


def test_a_window_starting_on_the_basetime_is_one_piece_held_outright() -> None:
    """(00,06] is a(0,6) itself; a max over it would be the 6h total."""
    (subwindow,) = _cut(CUMULATIVE, SIX)
    assert subwindow.is_direct
    with pytest.raises(ValueError, match="single piece"):
        check("max", "accum", Pieces.of_subwindows([[subwindow]]))


@pytest.mark.parametrize("valid_date", [SIX, NOON])
def test_over_cuts_a_cumulative_archive_into_hourly_totals(valid_date) -> None:
    subwindows = _cut(CUMULATIVE, valid_date, over=H)
    assert _hours(subwindows) == [1] * 6
    check("max", "accum", Pieces.of_subwindows([subwindows]))


def test_over_retrieves_each_field_once() -> None:
    """Neighbouring subwindows share an endpoint: a(0,7) closes one and opens the next."""
    subwindows = _cut(CUMULATIVE, NOON, over=H)
    assert len(contributions_of(subwindows)) == 12
    target = (NOON, None)
    assert len(argument([target], {target: subwindows}, PERIOD, forecast=False).intervals) == 7


def test_over_longer_than_the_archive_step_sums_the_fields_in_each_slice() -> None:
    """On an hourly archive ``over: 3h`` gives 3-hourly totals, not hourly ones."""
    subwindows = _cut({"accumulation": "1h"}, SIX, over=3 * H)
    assert _hours(subwindows) == [3, 3]
    assert [len(s.contributions) for s in subwindows] == [3, 3]


def test_over_on_a_stored_extremum_archive_is_refused() -> None:
    """Slices of stored maxima would be summed; 'over:' is for additive fields only."""
    subwindows = _cut({"accumulation": "1h"}, SIX, over=3 * H)
    with pytest.raises(ValueError, match="not additive"):
        check("max", "max", Pieces.of_subwindows([subwindows]))


# ── the reducer ──────────────────────────────────────────────────────

BASE = datetime.datetime(2021, 1, 1, 0)


def _a(hours: int) -> SignedInterval:
    """The field accumulated from the run start to step *hours*."""
    return SignedInterval(start=BASE, end=BASE + hours * H, base=BASE)


def _differenced(start: int, end: int) -> Subwindow:
    return Subwindow(
        interval=SignedInterval(start=BASE + start * H, end=BASE + end * H),
        contributions=(_a(end), -_a(start)),
    )


#: a(0,6) = 1, a(0,7) = 3, a(0,8) = 4: hourly totals 2 then 1.
TOTALS = {6: 1.0, 7: 3.0, 8: 4.0}


@pytest.mark.parametrize("operation,expected", [("sum", 3.0), ("max", 2.0), ("min", 1.0), ("mean", 1.5)])
def test_one_field_serves_two_subwindows(operation: str, expected: float) -> None:
    """a(0,7) is + in [6,7] and - in [7,8], within the same window."""
    reducer = Reducer(
        BASE + 8 * H,
        period=2 * H,
        key=(("param", "tp"),),
        subwindows=[_differenced(6, 7), _differenced(7, 8)],
        operation=OPERATIONS[operation],
    )
    shared = {h: np.full(3, v) for h, v in TOTALS.items()}
    for hours in (7, 6, 8):
        assert reducer.compute(shared[hours], _a(hours))

    assert reducer.is_complete()
    assert reducer.fields_used == 3
    assert np.allclose(reducer.operation.finalize(reducer._reduced, reducer._total_weight), expected)
    assert np.allclose(shared[7], 3.0), "the shared array is never modified"


def test_a_mean_weights_subwindows_by_length() -> None:
    """2 over one hour then 6 over three hours is a mean of 5 over four hours, not 4."""
    one = SignedInterval(start=BASE, end=BASE + H, base=BASE)
    three = SignedInterval(start=BASE + H, end=BASE + 4 * H, base=BASE)
    reducer = Reducer(
        BASE + 4 * H,
        period=4 * H,
        key=(("param", "avg_sdswrf"),),
        subwindows=[Subwindow(one, (one,)), Subwindow(three, (three,))],
        operation=OPERATIONS["mean"],
    )
    reducer.compute(np.full(2, 2.0), one)
    reducer.compute(np.full(2, 6.0), three)
    assert np.allclose(reducer.operation.finalize(reducer._reduced, reducer._total_weight), 5.0)


def test_a_half_filled_window_is_an_error() -> None:
    """A differenced subwindow yields nothing until both fields arrive; half is not dropped."""
    reducer = Reducer(
        BASE + 12 * H,
        period=PERIOD,
        key=(("param", "tp"),),
        subwindows=[_differenced(6, 12)],
        operation=OPERATIONS["sum"],
    )
    reducer.compute(np.ones(4), _a(12))
    assert reducer.fields_used == 1 and not reducer.is_complete()

    with pytest.raises(ValueError, match="missing source fields"):
        check_complete("accumulate", PERIOD, {(BASE + 12 * H, None, reducer.key): reducer}, [], [(BASE + 12 * H, None)])
