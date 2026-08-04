# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for non-additive reductions (max/min) in the accumulate source."""

import datetime

import numpy as np
import pytest

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate.accumulator import Accumulator
from anemoi.datasets.create.sources.accumulate.covering import ForecastCovering
from anemoi.datasets.create.sources.accumulate.covering import covering_factory
from anemoi.datasets.create.sources.accumulate.covering import validate_tiling
from anemoi.datasets.create.sources.accumulate.reductions import Max
from anemoi.datasets.create.sources.accumulate.reductions import Min
from anemoi.datasets.create.sources.accumulate.reductions import Sum
from anemoi.datasets.create.sources.accumulate.reductions import reduction_factory


def _hours(n):
    return datetime.timedelta(hours=n)


BASE = datetime.datetime(2021, 1, 1, 0)


class FakeTemplate:
    """Minimal stand-in for an earthkit field, for template-edition checks."""

    def __init__(self, edition):
        self._edition = edition

    def metadata(self, key):
        assert key == "edition", key
        return self._edition


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_factory_defaults_to_sum():
    assert isinstance(reduction_factory(None), Sum)
    assert isinstance(reduction_factory("sum"), Sum)


def test_factory_known_names():
    assert isinstance(reduction_factory("max"), Max)
    assert isinstance(reduction_factory("min"), Min)


def test_factory_passes_through_instances():
    r = Max()
    assert reduction_factory(r) is r


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown reduction 'mean'"):
        reduction_factory("mean")


def test_invertibility_and_step_types():
    assert Sum().invertible and Sum().grib_step_type == "accum"
    assert not Max().invertible and Max().grib_step_type == "max"
    assert not Min().invertible and Min().grib_step_type == "min"


# ---------------------------------------------------------------------------
# combine()
# ---------------------------------------------------------------------------


def test_sum_combine_adds_and_subtracts():
    r = Sum()
    acc = r.combine(None, np.array([1.0, 2.0]), 1)
    acc = r.combine(acc, np.array([10.0, 20.0]), 1)
    acc = r.combine(acc, np.array([1.0, 1.0]), -1)
    assert np.array_equal(acc, [10.0, 21.0])


def test_max_combine_takes_elementwise_maximum():
    r = Max()
    acc = r.combine(None, np.array([1.0, 5.0, 3.0]), 1)
    acc = r.combine(acc, np.array([4.0, 2.0, 3.0]), 1)
    assert np.array_equal(acc, [4.0, 5.0, 3.0])


def test_min_combine_takes_elementwise_minimum():
    r = Min()
    acc = r.combine(None, np.array([1.0, 5.0, 3.0]), 1)
    acc = r.combine(acc, np.array([4.0, 2.0, 3.0]), 1)
    assert np.array_equal(acc, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("reduction", [Sum(), Max(), Min()])
def test_combine_never_mutates_the_shared_values_array(reduction):
    """`values` is shared between every accumulator a field feeds."""
    shared = np.array([1.0, 2.0, 3.0])
    original = shared.copy()

    acc = reduction.combine(None, shared, 1)
    reduction.combine(acc, np.array([9.0, 0.0, 9.0]), 1)

    assert np.array_equal(shared, original), "the contributing field's values were modified"


@pytest.mark.parametrize("reduction", [Max(), Min()])
def test_extrema_propagate_nans_like_sum(reduction):
    acc = reduction.combine(None, np.array([1.0, np.nan]), 1)
    acc = reduction.combine(acc, np.array([2.0, 2.0]), 1)
    assert np.isnan(acc[1])


def test_extrema_reject_reversed_intervals():
    with pytest.raises(AssertionError):
        Max().combine(np.array([1.0]), np.array([2.0]), -1)


# ---------------------------------------------------------------------------
# GRIB encoding guards
# ---------------------------------------------------------------------------


def test_min_rejects_grib1_template():
    """GRIB1 shares timeRangeIndicator=2 between min and max, so min would read back as max."""
    with pytest.raises(ValueError, match="cannot be encoded in GRIB edition 1"):
        Min().check_template(FakeTemplate(1))


def test_min_accepts_grib2_template():
    Min().check_template(FakeTemplate(2))


@pytest.mark.parametrize("edition", [1, 2])
def test_sum_and_max_accept_any_edition(edition):
    Sum().check_template(FakeTemplate(edition))
    Max().check_template(FakeTemplate(edition))


# ---------------------------------------------------------------------------
# Covering constraints
# ---------------------------------------------------------------------------


def test_validate_tiling_accepts_contiguous_positive_cover():
    intervals = [SignedInterval(BASE + _hours(i), BASE + _hours(i + 1), base=BASE) for i in range(6)]
    validate_tiling(intervals, BASE, BASE + _hours(6))


def test_validate_tiling_rejects_gap():
    intervals = [
        SignedInterval(BASE, BASE + _hours(2), base=BASE),
        SignedInterval(BASE + _hours(3), BASE + _hours(6), base=BASE),
    ]
    with pytest.raises(ValueError, match="gap"):
        validate_tiling(intervals, BASE, BASE + _hours(6))


def test_validate_tiling_rejects_overlap():
    intervals = [
        SignedInterval(BASE, BASE + _hours(4), base=BASE),
        SignedInterval(BASE + _hours(3), BASE + _hours(6), base=BASE),
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_tiling(intervals, BASE, BASE + _hours(6))


def test_validate_tiling_rejects_overhang():
    intervals = [SignedInterval(BASE, BASE + _hours(9), base=BASE)]
    with pytest.raises(ValueError, match="does not line up"):
        validate_tiling(intervals, BASE, BASE + _hours(6))


def test_validate_tiling_rejects_reversed_interval():
    intervals = [
        SignedInterval(BASE, BASE + _hours(12), base=BASE),
        SignedInterval(BASE + _hours(6), BASE, base=BASE),
    ]
    with pytest.raises(ValueError, match="reversed"):
        validate_tiling(intervals, BASE + _hours(6), BASE + _hours(12))


def test_forecast_covering_rejects_from_zero_for_extrema():
    with pytest.raises(ValueError, match="from-zero"):
        ForecastCovering(period=_hours(6), accumulation="from-zero", positive_only=True)


def test_forecast_covering_allows_from_previous_step_for_extrema():
    covering = ForecastCovering(period=_hours(6), accumulation="from-previous-step", positive_only=True)
    cover = covering.cover(BASE + _hours(6), BASE + _hours(12), basetime=BASE)
    assert cover == [SignedInterval(start=BASE + _hours(6), end=BASE + _hours(12), base=BASE)]
    validate_tiling(cover, BASE + _hours(6), BASE + _hours(12))


def test_auto_covering_positive_only_tiles_hourly_archive():
    """An hourly from-previous-step archive tiles a 6h window with six positive intervals."""
    availability = [(0, "/".join(f"{i}-{i+1}" for i in range(24)))]
    covering = covering_factory({"auto": availability}, positive_only=True)
    cover = list(covering.cover(BASE + _hours(6), BASE + _hours(12)))
    assert len(cover) == 6
    assert all(i.length > 0 for i in cover)
    validate_tiling(cover, BASE + _hours(6), BASE + _hours(12))


def test_auto_covering_positive_only_refuses_from_zero_archive():
    """A from-zero archive can only reach [6,12] by subtraction, so max has no cover."""
    availability = [(0, "/".join(f"0-{i}" for i in range(1, 25)))]
    signed = covering_factory({"auto": availability}, positive_only=False)
    assert len(list(signed.cover(BASE + _hours(6), BASE + _hours(12)))) == 2  # +a(0,12) -a(0,6)

    positive = covering_factory({"auto": availability}, positive_only=True)
    with pytest.raises(ValueError, match="No forward-only covering exists"):
        positive.cover(BASE + _hours(6), BASE + _hours(12))


# ---------------------------------------------------------------------------
# End to end through Accumulator
# ---------------------------------------------------------------------------


def test_accumulator_max_over_six_hourly_fields():
    """The wind-gust case: 6h max out of six hourly maxima."""
    valid_date = BASE + _hours(6)
    coverage = [SignedInterval(BASE + _hours(i), BASE + _hours(i + 1), base=BASE) for i in range(6)]
    acc = Accumulator(valid_date, period=_hours(6), key=(("param", "10fg"),), coverage=coverage, reduction="max")

    # hour i has its peak gust in column i
    fields = []
    for i in range(6):
        values = np.zeros(6)
        values[i] = 10.0 + i
        fields.append(values)

    for values, interval in zip(fields, coverage):
        assert acc.compute(values, interval) is True

    assert acc.is_complete()
    assert np.array_equal(acc.values, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0])


def test_accumulator_sum_is_unchanged_by_default():
    """Regression guard: the default path still adds and subtracts."""
    valid_date = BASE + _hours(12)
    coverage = [
        SignedInterval(BASE, BASE + _hours(12), base=BASE),
        -SignedInterval(BASE, BASE + _hours(6), base=BASE),
    ]
    acc = Accumulator(valid_date, period=_hours(6), key=(("param", "tp"),), coverage=coverage)

    acc.compute(np.array([12.0, 12.0]), coverage[0])
    acc.compute(np.array([5.0, 4.0]), SignedInterval(BASE, BASE + _hours(6), base=BASE))

    assert acc.is_complete()
    assert np.array_equal(acc.values, [7.0, 8.0])


def test_accumulator_ignores_intervals_it_does_not_need():
    coverage = [SignedInterval(BASE, BASE + _hours(1), base=BASE)]
    acc = Accumulator(BASE + _hours(1), period=_hours(1), key=(("param", "10fg"),), coverage=coverage, reduction="max")

    unrelated = SignedInterval(BASE + _hours(5), BASE + _hours(6), base=BASE)
    assert acc.compute(np.array([1.0]), unrelated) is False
    assert not acc.is_complete()
