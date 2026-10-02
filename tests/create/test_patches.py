# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the metadata patches applied to fields before accumulation."""

import datetime

import pytest

from anemoi.datasets.create.sources.accumulate.covering import ForecastCovering
from anemoi.datasets.create.sources.accumulate.covering import covering_factory
from anemoi.datasets.create.sources.accumulate.covering import declared_step_ranges
from anemoi.datasets.create.sources.accumulate.field_to_interval import FieldToInterval

# The rr/se-al-ec layout: hourly to step 6, then 3-hourly.
RR_STEPS = "0-1/1-2/2-3/3-4/4-5/5-6/6-9/9-12/12-15/15-18/18-21/21-24"
RR_COVERING = {"auto": [[0, RR_STEPS], [12, RR_STEPS]]}


def _hours(n):
    return datetime.timedelta(hours=n)


class FakeField:
    """Minimal stand-in for a GRIB field, carrying only what FieldToInterval reads."""

    def __init__(self, date, time, startStep, endStep, validityDate, validityTime):
        self._md = dict(
            date=date,
            time=time,
            startStep=startStep,
            endStep=endStep,
            validityDate=validityDate,
            validityTime=validityTime,
        )

    def metadata(self, key):
        return self._md[key]

    def __repr__(self):
        return f"FakeField({self._md})"


def rr_gust_field(end_step, validity_time):
    """A field as rr/se-al-ec actually encodes it: startStep=0 whatever the real window."""
    return FakeField(
        date=20201230,
        time=1200,
        startStep=0,
        endStep=end_step,
        validityDate=20201230 if validity_time < 2400 else 20201231,
        validityTime=validity_time,
    )


# ---------------------------------------------------------------------------
# Reading the declared steps off a covering
# ---------------------------------------------------------------------------


def test_declared_step_ranges_maps_end_to_start():
    """Steps are timedeltas, so sub-hourly coverings work the same way."""
    mapping = declared_step_ranges(covering_factory(RR_COVERING))
    assert mapping[_hours(9)] == _hours(6)
    assert mapping[_hours(12)] == _hours(9)
    assert mapping[_hours(1)] == datetime.timedelta(0)
    assert mapping[_hours(6)] == _hours(5)


def test_declared_step_ranges_handles_sub_hourly_steps():
    mapping = declared_step_ranges(covering_factory({"auto": [[0, "0m-10m/10m-20m"]]}))
    assert mapping[datetime.timedelta(minutes=10)] == datetime.timedelta(0)
    assert mapping[datetime.timedelta(minutes=20)] == datetime.timedelta(minutes=10)


def test_declared_step_ranges_rejects_ambiguous_declaration():
    """One end step declared with two different start steps has no single window."""
    covering = covering_factory({"auto": [[0, "0-9"], [12, "6-9"]]})
    with pytest.raises(ValueError, match="two different start steps"):
        declared_step_ranges(covering)


def test_declared_step_ranges_needs_an_explicit_step_list():
    """A covering with no availability (e.g. the trajectory one) cannot supply the mapping."""
    covering = ForecastCovering(period=datetime.timedelta(hours=6), accumulation="from-previous-step")
    with pytest.raises(ValueError, match="explicit step list"):
        declared_step_ranges(covering)


# ---------------------------------------------------------------------------
# The patch itself
# ---------------------------------------------------------------------------


def test_patch_recovers_the_real_window():
    """The failing real case: step 9 is encoded 0-9 but is really the max over [6, 9]."""
    f2i = FieldToInterval(["start_step_from_covering"])
    assert f2i.needs_declared_steps
    f2i.bind_steps(declared_step_ranges(covering_factory(RR_COVERING)))

    interval = f2i(rr_gust_field(end_step=9, validity_time=2100))

    assert interval.start == datetime.datetime(2020, 12, 30, 18)
    assert interval.end == datetime.datetime(2020, 12, 30, 21)
    assert interval.base == datetime.datetime(2020, 12, 30, 12)


def test_patch_recovers_the_window_spanning_midnight():
    f2i = FieldToInterval(["start_step_from_covering"])
    f2i.bind_steps(declared_step_ranges(covering_factory(RR_COVERING)))

    # validityDate rolls over to the 31st for step 12 from a 12Z base
    interval = f2i(FakeField(20201230, 1200, 0, 12, 20201231, 0))

    assert interval.start == datetime.datetime(2020, 12, 30, 21)
    assert interval.end == datetime.datetime(2020, 12, 31, 0)


def test_patch_without_the_patch_leaves_the_wrong_window():
    """Regression guard: this is the behaviour that produced 'Field not used'."""
    f2i = FieldToInterval()
    assert not f2i.needs_declared_steps

    interval = f2i(rr_gust_field(end_step=9, validity_time=2100))

    assert interval.start == datetime.datetime(2020, 12, 30, 12)  # 0-9, not 6-9
    assert interval.end == datetime.datetime(2020, 12, 30, 21)


def test_patch_rejects_undeclared_end_step():
    f2i = FieldToInterval(["start_step_from_covering"])
    f2i.bind_steps(declared_step_ranges(covering_factory(RR_COVERING)))

    with pytest.raises(ValueError, match="declares no interval ending at step 7"):
        f2i(rr_gust_field(end_step=7, validity_time=1900))


def test_patch_without_bound_steps_is_an_error():
    f2i = FieldToInterval(["start_step_from_covering"])
    with pytest.raises(ValueError, match="needs the step ranges declared in 'covering:'"):
        f2i(rr_gust_field(end_step=9, validity_time=2100))


# ---------------------------------------------------------------------------
# The pre-existing patches keep working
# ---------------------------------------------------------------------------


def test_set_start_step_to_zero_still_works():
    f2i = FieldToInterval(["set_start_step_to_zero"])
    interval = f2i(FakeField(20201230, 0, 6, 9, 20201230, 900))
    assert interval.start == datetime.datetime(2020, 12, 30, 0)
    assert interval.end == datetime.datetime(2020, 12, 30, 9)


def test_reset_24h_accumulations_still_works():
    f2i = FieldToInterval(["reset_24h_accumulations"])
    interval = f2i(FakeField(20201230, 0, 0, 27, 20201231, 300))
    assert interval.start == datetime.datetime(2020, 12, 31, 0)  # 27 -> 24-27
    assert interval.end == datetime.datetime(2020, 12, 31, 3)


def test_unknown_patch_is_rejected():
    with pytest.raises(ValueError, match="Unknown patch key"):
        FieldToInterval(["no_such_patch"])
