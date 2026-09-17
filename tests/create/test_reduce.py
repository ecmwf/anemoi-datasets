# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the reduce source and the operations it folds windows with."""

import datetime

import numpy as np
import pytest

from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate.accumulator import Accumulator
from anemoi.datasets.create.sources.accumulate.covering import ForecastCovering
from anemoi.datasets.create.sources.accumulate.covering import covering_factory
from anemoi.datasets.create.sources.accumulate.covering import validate_tiling
from anemoi.datasets.create.sources.accumulate.operations import Max
from anemoi.datasets.create.sources.accumulate.operations import Mean
from anemoi.datasets.create.sources.accumulate.operations import Min
from anemoi.datasets.create.sources.accumulate.operations import Sum
from anemoi.datasets.create.sources.accumulate.operations import operation_factory


def _hours(n):
    return datetime.timedelta(hours=n)


BASE = datetime.datetime(2021, 1, 1, 0)


class FakeTemplate:
    """Minimal stand-in for an earthkit field, for GRIB-encoding checks.

    ``step_type_for_conversion`` is the param-derived key eccodes exposes on
    edition 1 (10fg -> "max", mn2t6 -> "min", tp -> "accum", 2t -> "unknown").
    """

    def __init__(self, edition, step_type_for_conversion=None):
        self._md = {"edition": edition, "stepTypeForConversion": step_type_for_conversion}

    def metadata(self, key, default=None):
        return self._md.get(key, default)


# ---------------------------------------------------------------------------
# The operation registry
# ---------------------------------------------------------------------------


def test_factory_defaults_to_sum():
    assert isinstance(operation_factory(None), Sum)
    assert isinstance(operation_factory("sum"), Sum)


def test_factory_known_names():
    assert isinstance(operation_factory("max"), Max)
    assert isinstance(operation_factory("min"), Min)


def test_factory_passes_through_instances():
    r = Max()
    assert operation_factory(r) is r


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown operation 'median'"):
        operation_factory("median")


def test_invertibility_and_step_types():
    assert Sum().invertible and Sum().grib_step_type == "accum"
    assert not Max().invertible and Max().grib_step_type == "max"
    assert not Min().invertible and Min().grib_step_type == "min"
    assert not Mean().invertible and Mean().grib_step_type == "avg"


# ---------------------------------------------------------------------------
# combine()
# ---------------------------------------------------------------------------


def test_sum_combine_adds_and_subtracts():
    r = Sum()
    acc = r.combine(None, np.array([1.0, 2.0]), 1, 3600.0)
    acc = r.combine(acc, np.array([10.0, 20.0]), 1, 3600.0)
    acc = r.combine(acc, np.array([1.0, 1.0]), -1, 3600.0)
    assert np.array_equal(acc, [10.0, 21.0])


def test_max_combine_takes_elementwise_maximum():
    r = Max()
    acc = r.combine(None, np.array([1.0, 5.0, 3.0]), 1, 3600.0)
    acc = r.combine(acc, np.array([4.0, 2.0, 3.0]), 1, 3600.0)
    assert np.array_equal(acc, [4.0, 5.0, 3.0])


def test_min_combine_takes_elementwise_minimum():
    r = Min()
    acc = r.combine(None, np.array([1.0, 5.0, 3.0]), 1, 3600.0)
    acc = r.combine(acc, np.array([4.0, 2.0, 3.0]), 1, 3600.0)
    assert np.array_equal(acc, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("operation", [Sum(), Max(), Min(), Mean()])
def test_combine_never_mutates_the_shared_values_array(operation):
    """`values` is shared between every accumulator a field feeds."""
    shared = np.array([1.0, 2.0, 3.0])
    original = shared.copy()

    acc = operation.combine(None, shared, 1, 3600.0)
    operation.combine(acc, np.array([9.0, 0.0, 9.0]), 1, 3600.0)

    assert np.array_equal(shared, original), "the contributing field's values were modified"


@pytest.mark.parametrize("operation", [Max(), Min(), Mean()])
def test_extrema_propagate_nans_like_sum(operation):
    acc = operation.combine(None, np.array([1.0, np.nan]), 1, 3600.0)
    acc = operation.combine(acc, np.array([2.0, 2.0]), 1, 3600.0)
    assert np.isnan(acc[1])


def test_extrema_reject_reversed_intervals():
    with pytest.raises(AssertionError):
        Max().combine(np.array([1.0]), np.array([2.0]), -1, 3600.0)


# ---------------------------------------------------------------------------
# GRIB encoding guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", [Max(), Min(), Sum(), Mean()])
def test_grib2_needs_no_promotion(operation):
    """Edition 2 carries the statistic in the message, so nothing is ambiguous."""
    assert operation.output_edition(FakeTemplate(2)) is None


@pytest.mark.parametrize("operation", [Sum(), Mean()])
@pytest.mark.parametrize("declared", ["max", "accum", "unknown", None])
def test_grib1_keeps_sum_and_mean(operation, declared):
    """timeRangeIndicator has codes for accumulation (4) and average (3)."""
    assert operation.output_edition(FakeTemplate(1, declared)) is None


def test_grib1_keeps_an_extremum_the_parameter_declares():
    """10fg declares itself a maximum, so a max is expressible in edition 1."""
    assert Max().output_edition(FakeTemplate(1, "max")) is None
    assert Min().output_edition(FakeTemplate(1, "min")) is None  # e.g. mn2t6


def test_grib1_promotes_min_on_a_max_parameter():
    """Would otherwise be silently recorded as process=maximum."""
    assert Min().output_edition(FakeTemplate(1, "max")) == 2


@pytest.mark.parametrize("declared", ["accum", "unknown", None])
def test_grib1_promotes_an_extremum_the_parameter_does_not_declare(declared):
    """max on tp/2t would otherwise resolve to nothing and fail later in the build."""
    assert Max().output_edition(FakeTemplate(1, declared)) == 2


def test_promotion_warns_once(caplog):
    operation = Max()
    with caplog.at_level("WARNING"):
        for _ in range(3):
            operation.output_edition(FakeTemplate(1, "accum"))
    warnings = [r for r in caplog.records if "GRIB2" in r.getMessage()]
    assert len(warnings) == 1
    assert "rename:" in warnings[0].getMessage()


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


def test_forecast_covering_rejects_from_zero_when_positive_only():
    with pytest.raises(ValueError, match="from-zero"):
        ForecastCovering(period=_hours(6), accumulation="from-zero", positive_only=True)


def test_forecast_covering_allows_from_previous_step_when_positive_only():
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
    acc = Accumulator(valid_date, period=_hours(6), key=(("param", "10fg"),), coverage=coverage, operation="max")

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
    acc = Accumulator(BASE + _hours(1), period=_hours(1), key=(("param", "10fg"),), coverage=coverage, operation="max")

    unrelated = SignedInterval(BASE + _hours(5), BASE + _hours(6), base=BASE)
    assert acc.compute(np.array([1.0]), unrelated) is False
    assert not acc.is_complete()


# ---------------------------------------------------------------------------
# The reduce source itself
# ---------------------------------------------------------------------------

HOURLY = [[0, "/".join(f"{i}-{i+1}" for i in range(24))], [12, "/".join(f"{i}-{i+1}" for i in range(24))]]
FROM_ZERO = [[0, "/".join(f"0-{i}" for i in range(1, 25))]]


class _RecordingAction:
    """Stands in for the inner source; records the intervals it was asked for."""

    def __init__(self, requested):
        self.requested = requested

    def __call__(self, context, argument):
        for i in argument.intervals:
            self.requested.append((i.base, int((i.max - i.base).total_seconds() / 3600), i.sign))
        return []


class _FakeContext:
    def __init__(self):
        self.requested = []

    def create_source(self, config, *path):
        return _RecordingAction(self.requested)

    def trace(self, *args, **kwargs):
        pass


def build_reduce(**kwargs):
    from anemoi.datasets.create.sources.reduce.source import ReduceSource

    config = dict(
        period="6h",
        covering={"auto": HOURLY},
        source={"mars": {"class": "od", "param": ["10fg"], "type": "fc", "levtype": "sfc", "stream": "oper"}},
    )
    config.update(kwargs)
    return ReduceSource(context=_FakeContext(), **config)


def test_reduce_defaults_to_sum():
    assert isinstance(build_reduce().operation, Sum)


def test_reduce_accepts_the_operation_key():
    assert isinstance(build_reduce(operation="max").operation, Max)
    assert isinstance(build_reduce(operation="min").operation, Min)


def test_reduce_rejects_unknown_operation():
    with pytest.raises(ValueError, match="Unknown operation"):
        build_reduce(operation="median")


def test_reduce_rejects_unsupported_source():
    with pytest.raises(ValueError, match="not supported by 'reduce'"):
        build_reduce(source={"netcdf": {"path": "/tmp/x.nc"}})


@pytest.mark.parametrize("source_name", ["mars", "fdb", "grib-index"])
def test_reduce_accepts_the_documented_sources(source_name):
    assert build_reduce(source={source_name: {"param": ["10fg"]}})._source_name == source_name


def test_reduce_never_subtracts_even_for_sum():
    """The scope rule: reduce aggregates a tiling, it does not de-aggregate."""
    for operation in ("sum", "max", "min"):
        src = build_reduce(operation=operation)
        assert src.POSITIVE_ONLY is True
        try:
            src.execute_valid_dates(ValidDates([datetime.datetime(2021, 1, 1, 12)]))
        except ValueError as e:
            assert "No accumulators" in str(e)  # the fake source returns no fields
        requested = src.context.requested
        assert requested, operation
        assert all(sign > 0 for _, _, sign in requested), operation
        assert len({r for r in requested}) == 6, operation  # six hourly intervals per 6h window


@pytest.mark.parametrize("operation", ["sum", "max", "min"])
def test_reduce_refuses_an_accumulated_from_start_archive(operation):
    """The documented sharp edge: de-aggregation is out of scope, so fail early."""
    src = build_reduce(operation=operation, covering={"auto": FROM_ZERO})
    with pytest.raises(ValueError, match="No forward-only covering exists"):
        src.execute_valid_dates(ValidDates([datetime.datetime(2021, 1, 1, 12)]))


def test_reduce_rejects_trajectory_recipes():
    from anemoi.datasets.create.arguments import ForecastDates

    src = build_reduce(operation="max")
    with pytest.raises(NotImplementedError, match="does not support trajectory recipes"):
        src.execute_forecast_dates(ForecastDates([(datetime.datetime(2021, 1, 1, 6), datetime.datetime(2021, 1, 1))]))


def test_accumulate_is_unchanged_and_still_subtracts():
    """Regression guard: `accumulate` keeps its signed decomposition and has no operation key."""
    import inspect

    from anemoi.datasets.create.sources.accumulate.source import AccumulateSource

    assert AccumulateSource.POSITIVE_ONLY is False
    params = inspect.signature(AccumulateSource.__init__).parameters
    assert "operation" not in params and "reduction" not in params

    src = AccumulateSource(
        context=_FakeContext(),
        period="6h",
        covering={"auto": FROM_ZERO},
        source={"mars": {"class": "od", "param": ["tp"], "type": "fc", "levtype": "sfc", "stream": "oper"}},
    )
    assert isinstance(src.operation, Sum)
    try:
        src.execute_valid_dates(ValidDates([datetime.datetime(2021, 1, 1, 12)]))
    except ValueError as e:
        assert "No accumulators" in str(e)
    # +a(0,12) - a(0,6): accumulate still reaches the window by subtraction
    assert any(sign < 0 for _, _, sign in src.context.requested)


# ---------------------------------------------------------------------------
# mean: the length-weighted average
# ---------------------------------------------------------------------------


def test_mean_is_registered_under_both_spellings():
    assert isinstance(operation_factory("mean"), Mean)
    assert isinstance(operation_factory("avg"), Mean)
    assert isinstance(operation_factory("average"), Mean)
    assert Mean().grib_step_type == "avg"
    assert not Mean().invertible


def test_mean_of_equal_intervals_is_the_plain_average():
    op = Mean()
    hour = 3600.0
    acc = op.combine(None, np.array([1.0, 10.0]), 1, hour)
    acc = op.combine(acc, np.array([2.0, 20.0]), 1, hour)
    acc = op.combine(acc, np.array([3.0, 30.0]), 1, hour)
    assert np.allclose(op.finalize(acc, 3 * hour), [2.0, 20.0])


def test_mean_weights_by_interval_length():
    """A 1h piece and a 3h piece must not count equally."""
    op = Mean()
    acc = op.combine(None, np.array([10.0]), 1, 3600.0)       # 1h at 10
    acc = op.combine(acc, np.array([2.0]), 1, 3 * 3600.0)     # 3h at 2
    # (1*10 + 3*2) / 4 = 4.0, not the unweighted (10+2)/2 = 6.0
    assert np.allclose(op.finalize(acc, 4 * 3600.0), [4.0])


def test_mean_does_not_mutate_the_shared_values_array():
    op = Mean()
    shared = np.array([1.0, 2.0, 3.0])
    original = shared.copy()
    acc = op.combine(None, shared, 1, 3600.0)
    op.combine(acc, np.array([9.0, 9.0, 9.0]), 1, 3600.0)
    assert np.array_equal(shared, original)


def test_mean_rejects_reversed_intervals():
    with pytest.raises(AssertionError):
        Mean().combine(np.array([1.0]), np.array([2.0]), -1, 3600.0)


@pytest.mark.parametrize("operation", [Sum(), Max(), Min()])
def test_other_operations_ignore_the_weight(operation):
    """Only Mean is length-weighted; the rest must be unaffected by it."""
    a = operation.combine(None, np.array([2.0, 4.0]), 1, 3600.0)
    b = operation.combine(None, np.array([2.0, 4.0]), 1, 999999.0)
    assert np.array_equal(a, b)
    assert np.array_equal(operation.finalize(a, 12345.0), a)


def test_accumulator_mean_over_a_mixed_length_tiling():
    """The rr/se-al-ec shape: a 6h window tiled by 1h+1h+1h+3h pieces."""
    base = BASE
    coverage = [
        SignedInterval(base + _hours(0), base + _hours(1), base=base),
        SignedInterval(base + _hours(1), base + _hours(2), base=base),
        SignedInterval(base + _hours(2), base + _hours(3), base=base),
        SignedInterval(base + _hours(3), base + _hours(6), base=base),
    ]
    acc = Accumulator(
        base + _hours(6), period=_hours(6), key=(("param", "2t"),), coverage=coverage, operation="mean"
    )
    for values, interval in zip(
        [np.array([12.0]), np.array([6.0]), np.array([6.0]), np.array([2.0])], coverage
    ):
        assert acc.compute(values, interval) is True

    assert acc.is_complete()
    assert acc.total_weight == 6 * 3600.0
    # (1*12 + 1*6 + 1*6 + 3*2) / 6 = 5.0
    assert np.allclose(Mean().finalize(acc.values, acc.total_weight), [5.0])


def test_reduce_accepts_mean():
    src = build_reduce(operation="mean")
    assert isinstance(src.operation, Mean)
    assert src.POSITIVE_ONLY is True


# ---------------------------------------------------------------------------
# Instantaneous fields must be rejected, not silently treated as [0, endStep]
# ---------------------------------------------------------------------------


class InstantField:
    """An instantaneous analysis field: startStep == endStep."""

    def __init__(self, step=6, step_type="instant"):
        self._md = {
            "date": 20210101, "time": 0, "startStep": step, "endStep": step,
            "validityDate": 20210101, "validityTime": step * 100, "stepType": step_type,
        }

    def metadata(self, key):
        return self._md[key]

    def __repr__(self):
        return f"GribField(2t,step={self._md['endStep']},instant)"


def test_accumulate_still_reads_instant_fields_as_zero_to_endstep():
    """Regression guard: accumulate depends on this fallback for GRIB1 accumulations."""
    from anemoi.datasets.create.sources.accumulate.field_to_interval import FieldToInterval

    interval = FieldToInterval()(InstantField(step=6))
    assert interval.start == datetime.datetime(2021, 1, 1, 0)
    assert interval.end == datetime.datetime(2021, 1, 1, 6)


def test_require_interval_rejects_an_instantaneous_field():
    from anemoi.datasets.create.sources.accumulate.field_to_interval import FieldToInterval

    with pytest.raises(ValueError, match="covers no time interval"):
        FieldToInterval(require_interval=True)(InstantField(step=6))


def test_rejection_names_both_likely_causes():
    from anemoi.datasets.create.sources.accumulate.field_to_interval import FieldToInterval

    with pytest.raises(ValueError) as excinfo:
        FieldToInterval(require_interval=True)(InstantField(step=6))
    message = str(excinfo.value)
    assert "stepType='instant'" in message      # the evidence
    assert "instantaneous" in message           # cause 1: wrong parameter for this source
    assert "patch:" in message                  # cause 2: repairable metadata


def test_reduce_requires_real_intervals():
    assert build_reduce()._field_to_interval.require_interval is True


def test_accumulate_does_not_require_real_intervals():
    from anemoi.datasets.create.sources.accumulate.source import AccumulateSource

    src = AccumulateSource(
        context=_FakeContext(),
        period="6h",
        covering={"auto": HOURLY},
        source={"mars": {"class": "od", "param": ["tp"], "type": "fc", "levtype": "sfc", "stream": "oper"}},
    )
    assert src._field_to_interval.require_interval is False
