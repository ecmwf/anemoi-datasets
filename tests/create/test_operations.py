# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The two-level reduction: states make values, an operation reduces them."""

import datetime

import numpy as np
import pytest

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.windowed.operations import Max
from anemoi.datasets.create.sources.windowed.operations import Mean
from anemoi.datasets.create.sources.windowed.operations import Min
from anemoi.datasets.create.sources.windowed.operations import Sum
from anemoi.datasets.create.sources.windowed.operations import operation_factory
from anemoi.datasets.create.sources.windowed.reducer import Reducer
from anemoi.datasets.create.sources.windowed.samples import Sample
from anemoi.datasets.create.sources.windowed.states import SampleState
from anemoi.datasets.create.sources.windowed.states import SubwindowState
from anemoi.datasets.create.sources.windowed.subwindows import Subwindow
from anemoi.datasets.create.sources.windowed.subwindows import direct

BASE = datetime.datetime(2021, 1, 1, 0)


def _hours(n):
    return datetime.timedelta(hours=n)


def _interval(a, b):
    return SignedInterval(start=BASE + _hours(a), end=BASE + _hours(b), base=BASE)


def _reducer(states, operation="sum", period=_hours(6), param="tp"):
    return Reducer(BASE + period, period=period, key=(("param", param),), states=states, operation=operation)


# ── the operations ───────────────────────────────────────────────────


def test_factory_and_step_types():
    assert isinstance(operation_factory(None), Sum)
    assert (Sum().time_method, Max().time_method, Min().time_method, Mean().time_method) == (
        "accum",
        "max",
        "min",
        "avg",
    )


def test_factory_accepts_the_recipe_spellings():
    assert isinstance(operation_factory("maximum"), Max)
    assert isinstance(operation_factory("average"), Mean)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown operation 'median'"):
        operation_factory("median")


def test_only_sum_is_differenceable():
    """Differencing is a property of the stored statistic, not of the reduction."""
    assert Sum().differenceable
    assert not (Max().differenceable or Min().differenceable or Mean().differenceable)


@pytest.mark.parametrize("operation", [Max(), Min(), Mean()])
def test_non_differenceable_operations_explain_themselves(operation):
    assert operation.why_not_differenceable()


@pytest.mark.parametrize("operation", [Max(), Min(), Mean(), Sum()])
def test_reductions_propagate_nans(operation):
    reduced = operation.reduce(None, np.array([1.0, np.nan]), 1.0)
    reduced = operation.reduce(reduced, np.array([2.0, 2.0]), 1.0)
    assert np.isnan(reduced[1])


# ── the two levels, through the Reducer ──────────────────────────────


def test_max_over_direct_subwindows():
    """The wind-gust shape: six hourly maxima reduced to a 6h maximum."""
    states = [SubwindowState(direct(_interval(i, i + 1))) for i in range(6)]
    reducer = _reducer(states, operation="max", param="10fg")

    for i in range(6):
        values = np.zeros(6)
        values[i] = 10.0 + i
        assert reducer.compute(values, _interval(i, i + 1)) is True

    assert reducer.is_complete()
    assert np.array_equal(reducer.values, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])


def test_sum_over_a_differenced_subwindow():
    """accumulate's from-zero path: the signed sum happens inside the state."""
    subwindow = Subwindow(interval=_interval(6, 12), contributions=(_interval(0, 12), -_interval(0, 6)))
    reducer = _reducer([SubwindowState(subwindow)])

    reducer.compute(np.array([12.0, 12.0]), _interval(0, 12))
    assert reducer.values is None, "the subwindow was reduced before its second contribution arrived"

    reducer.compute(np.array([5.0, 4.0]), _interval(0, 6))
    assert reducer.is_complete()
    assert np.array_equal(reducer.values, [7.0, 8.0])


def test_a_field_can_fill_two_neighbouring_subwindows():
    """Cumulative subwindows share an endpoint field."""
    states = [
        SubwindowState(Subwindow(interval=_interval(6, 7), contributions=(_interval(0, 7), -_interval(0, 6)))),
        SubwindowState(Subwindow(interval=_interval(7, 8), contributions=(_interval(0, 8), -_interval(0, 7)))),
    ]
    reducer = _reducer(states, period=_hours(2))

    reducer.compute(np.array([6.0]), _interval(0, 6))
    reducer.compute(np.array([10.0]), _interval(0, 8))
    assert reducer.compute(np.array([9.0]), _interval(0, 7)) is True  # closes one, opens the other

    assert reducer.is_complete()
    assert np.array_equal(reducer.values, [(9.0 - 6.0) + (10.0 - 9.0)])


def test_mean_weighs_subwindows_by_length():
    """A mixed-granularity archive: two 1h subwindows and a 3h one."""
    states = [
        SubwindowState(direct(_interval(0, 1))),
        SubwindowState(direct(_interval(1, 2))),
        SubwindowState(direct(_interval(2, 5))),
    ]
    reducer = _reducer(states, operation="mean", period=_hours(5))

    reducer.compute(np.array([0.0]), _interval(0, 1))
    reducer.compute(np.array([0.0]), _interval(1, 2))
    reducer.compute(np.array([5.0]), _interval(2, 5))

    assert reducer.is_complete()
    # (1*0 + 1*0 + 3*5) / 5 == 3, not the unweighted 5/3
    assert np.array_equal(reducer.operation.finalize(reducer.values, reducer.total_weight), [3.0])


def test_mean_over_samples_is_the_plain_average():
    """Samples weigh 1 each, so a weighted mean over them is the arithmetic mean."""
    times = [BASE + _hours(i) for i in (1, 2)]
    reducer = _reducer([SampleState(Sample(t)) for t in times], operation="average", period=_hours(2), param="2t")

    reducer.compute(np.array([0.0]), (times[0], None))
    reducer.compute(np.array([4.0]), (times[1], None))

    assert np.array_equal(reducer.operation.finalize(reducer.values, reducer.total_weight), [2.0])


def test_a_field_offered_to_a_window_that_does_not_want_it():
    reducer = _reducer([SubwindowState(direct(_interval(0, 1)))], period=_hours(1))
    assert reducer.compute(np.zeros(2), _interval(5, 6)) is False


def test_a_repeat_after_completion_is_simply_not_needed():
    """Overlapping windows offer the same field once per window that wants it."""
    reducer = _reducer([SubwindowState(direct(_interval(0, 1)))], period=_hours(1))
    assert reducer.compute(np.zeros(2), _interval(0, 1)) is True
    assert reducer.compute(np.zeros(2), _interval(0, 1)) is False


@pytest.mark.parametrize("operation", ["sum", "max", "min", "mean"])
def test_the_shared_values_array_is_never_mutated(operation):
    """One field feeds many windows; its array is shared between them."""
    shared = np.array([1.0, 2.0, 3.0])
    original = shared.copy()

    reducer = _reducer([SubwindowState(direct(_interval(0, 1)))], operation=operation, period=_hours(1))
    reducer.compute(shared, _interval(0, 1))
    reducer.operation.finalize(reducer.values, reducer.total_weight)

    assert np.array_equal(shared, original)


# ── the guard: a non-additive field may not be differenced ───────────


@pytest.mark.parametrize("statistic", ["max", "min", "avg"])
def test_differencing_a_non_additive_field_is_rejected(statistic):
    """What `accumulate` on a wind-gust archive would otherwise compute silently."""
    subwindow = Subwindow(interval=_interval(6, 9), contributions=(_interval(0, 9), -_interval(0, 6)))
    reducer = _reducer([SubwindowState(subwindow)], period=_hours(3), param="10fg")

    with pytest.raises(ValueError, match="is not additive"):
        reducer.compute(np.array([1.0]), _interval(0, 9), statistic=statistic)


def test_differencing_an_accumulation_is_allowed():
    subwindow = Subwindow(interval=_interval(6, 12), contributions=(_interval(0, 12), -_interval(0, 6)))
    reducer = _reducer([SubwindowState(subwindow)])
    assert reducer.compute(np.array([1.0]), _interval(0, 12), statistic="accum") is True


@pytest.mark.parametrize("statistic", ["instant", None, "unknown"])
def test_an_ambiguous_statistic_is_left_alone(statistic):
    """GRIB1 stores accumulations with stepType=instant and says nothing."""
    subwindow = Subwindow(interval=_interval(6, 12), contributions=(_interval(0, 12), -_interval(0, 6)))
    reducer = _reducer([SubwindowState(subwindow)])
    assert reducer.compute(np.array([1.0]), _interval(0, 12), statistic=statistic) is True


@pytest.mark.parametrize("statistic", ["max", "min", "avg"])
def test_a_direct_subwindow_accepts_any_statistic(statistic):
    """A max field is exactly what a direct subwindow of a gust archive holds."""
    reducer = _reducer([SubwindowState(direct(_interval(0, 1)))], operation="max", period=_hours(1), param="10fg")
    assert reducer.compute(np.array([1.0]), _interval(0, 1), statistic=statistic) is True


# ── over: reducing what differencing produced ────────────────────────


def _hourly_from_cumulative(first_hour: int, hours: int):
    """Subwindows [h, h+1] rebuilt as +a(0,h+1) - a(0,h), as ``over: 1h`` produces."""
    return [
        Subwindow(interval=_interval(h, h + 1), contributions=(_interval(0, h + 1), -_interval(0, h)))
        for h in range(first_hour, first_hour + hours)
    ]


#: Cumulative totals at steps 1..5, i.e. hourly increments of 1, 6, 1, 1.
CUMULATIVE = {1: 2.0, 2: 3.0, 3: 9.0, 4: 10.0, 5: 11.0}


def _feed(reducer, statistic="accum"):
    for step, total in CUMULATIVE.items():
        reducer.compute(np.array([total]), _interval(0, step), statistic=statistic)


def test_max_of_differenced_subwindows_is_the_wettest_hour():
    """What ``over: 1h`` under ``maximum:`` computes, at the level it computes it.

    Neither the maximum of the cumulative fields (11) nor the window total (9) is the
    answer, which is the whole reason the subwindows exist.
    """
    reducer = _reducer(
        [SubwindowState(s) for s in _hourly_from_cumulative(1, 4)], operation="max", period=_hours(4)
    )
    _feed(reducer)

    assert reducer.is_complete()
    assert np.array_equal(reducer.values, [6.0]), "the wettest hour"


def test_the_same_subwindows_summed_give_the_window_total():
    """Only the reduction differs: summing them re-accumulates the window."""
    reducer = _reducer(
        [SubwindowState(s) for s in _hourly_from_cumulative(1, 4)], operation="sum", period=_hours(4)
    )
    _feed(reducer)

    assert np.array_equal(reducer.values, [9.0]), "11 - 2, the total over [1,5]"


def test_a_non_additive_field_is_still_refused_with_over():
    """``over:`` says how long a subwindow is, not that differencing is safe.

    A gust archive differenced into hourly parts is meaningless however the parts
    are then reduced; the field's own statistic is what catches it.
    """
    reducer = _reducer(
        [SubwindowState(s) for s in _hourly_from_cumulative(1, 4)], operation="max", period=_hours(4)
    )
    with pytest.raises(ValueError, match="is not additive"):
        reducer.compute(np.array([1.0]), _interval(0, 2), statistic="max")


# ── a part carries its own weight ─────────────────────────────────────


def test_both_kinds_of_part_answer_weight():
    """The weight belongs to the part, which is what lets one Mean serve both.

    A subwindow weighs its own length, so a partition of mixed granularity is
    length-weighted; a sample weighs 1, so evenly-spaced instants give the plain
    arithmetic mean.
    """
    assert direct(_interval(2, 5)).weight == 3 * 3600.0
    assert Sample(BASE).weight == 1.0


def test_a_state_takes_its_weight_from_its_part():
    """Neither state decides for itself; both read it off the part."""
    subwindow = direct(_interval(2, 5))
    assert SubwindowState(subwindow).weight == subwindow.weight

    sample = Sample(BASE)
    assert SampleState(sample).weight == sample.weight
