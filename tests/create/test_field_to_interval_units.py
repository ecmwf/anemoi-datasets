# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Unit handling in ``FieldToInterval``.

Two ways a field states its accumulation window, both of which used to be read
as whole hours:

* GRIB keys ``startStep``/``endStep``, returned in the units of the message's
  ``stepUnits`` -- a minute-unit field reads back as the *string* ``"0m"`` /
  ``"10m"`` (the same eccodes behaviour the grib index documents);
* the earthkit ``time.step`` / ``proc.time_value`` components of an in-memory
  field, which are timedeltas (this is what ``accumulate:`` and the reduction
  sources emit, so it is the path taken when one feeds another).
"""

import datetime

import pytest

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate.field_to_interval import FieldToInterval


def _minutes(n):
    return datetime.timedelta(minutes=n)


class _GribField:
    """A field exposing only GRIB metadata keys, as a file-backed field does."""

    def __init__(self, base, start_step, end_step, valid):
        self._meta = {
            "date": base.strftime("%Y%m%d"),
            "time": base.strftime("%H%M"),
            "startStep": start_step,
            "endStep": end_step,
            "validityDate": valid.strftime("%Y%m%d"),
            "validityTime": valid.strftime("%H%M"),
        }

    def metadata(self, key):
        return self._meta[key]


class _Component:
    def __init__(self, value):
        self._value = value

    def __call__(self):
        return self._value


class _Grib1MinuteField(_GribField):
    """A GRIB1 message whose minute-unit step cannot be read back in hours.

    eccodes hands back the raw number (``10`` for 10 minutes), which reads as
    10 *hours*; the validity time is the giveaway.
    """

    def __init__(self, base, start_step, end_step, valid):
        super().__init__(base, start_step, end_step, valid)
        self._meta["indicatorOfUnitOfTimeRange"] = 0
        self._meta["edition"] = 1


class _MemoryField:
    """A field exposing only the earthkit time/proc components."""

    def __init__(self, base, step, length):
        self.time = type("T", (), {"base_datetime": _Component(base), "step": _Component(step)})()
        self.proc = type("P", (), {"time_value": _Component(length)})()

    def metadata(self, key):
        raise KeyError(key)


BASE = datetime.datetime(2020, 10, 1, 0)


def test_grib_minute_unit_steps_are_strings():
    """``startStep``/``endStep`` of a minute-unit message read back as "0m"/"10m"."""
    field = _GribField(BASE, "0m", "10m", BASE + _minutes(10))
    interval = FieldToInterval()(field)
    assert interval == SignedInterval(start=BASE, end=BASE + _minutes(10), base=BASE)


def test_grib_minute_unit_non_zero_start():
    field = _GribField(BASE, "10m", "20m", BASE + _minutes(20))
    interval = FieldToInterval()(field)
    assert interval == SignedInterval(start=BASE + _minutes(10), end=BASE + _minutes(20), base=BASE)


def test_grib_compound_step_syntax():
    """A step past the hour is spelled "1h10m" by eccodes when the unit is minutes."""
    field = _GribField(BASE, "1h", "1h10m", BASE + _minutes(70))
    interval = FieldToInterval()(field)
    assert interval == SignedInterval(start=BASE + _minutes(60), end=BASE + _minutes(70), base=BASE)


def test_grib_hourly_integer_steps_still_work():
    """The ordinary case: integer hours, unchanged."""
    field = _GribField(BASE, 6, 12, BASE + datetime.timedelta(hours=12))
    interval = FieldToInterval()(field)
    assert interval == SignedInterval(
        start=BASE + datetime.timedelta(hours=6),
        end=BASE + datetime.timedelta(hours=12),
        base=BASE,
    )


def test_grib_equal_steps_are_read_as_from_zero():
    """The built-in normalisation (startStep == endStep means 0-endStep) is unit-aware."""
    field = _GribField(BASE, "10m", "10m", BASE + _minutes(10))
    interval = FieldToInterval()(field)
    assert interval == SignedInterval(start=BASE, end=BASE + _minutes(10), base=BASE)


def test_in_memory_sub_hourly_window():
    """A 10-minute in-memory accumulation used to floor to a zero-length window."""
    field = _MemoryField(BASE, _minutes(10), _minutes(10))
    interval = FieldToInterval()(field)
    assert interval == SignedInterval(start=BASE, end=BASE + _minutes(10), base=BASE)


def test_in_memory_sub_hourly_window_with_offset():
    field = _MemoryField(BASE, _minutes(50), _minutes(20))
    interval = FieldToInterval()(field)
    assert interval == SignedInterval(start=BASE + _minutes(30), end=BASE + _minutes(50), base=BASE)


def test_in_memory_without_length_is_refused():
    field = _MemoryField(BASE, _minutes(10), None)
    with pytest.raises(ValueError, match="proc.time_value"):
        FieldToInterval()(field)


def test_reset_24h_patch_keeps_sub_hourly_offsets():
    """``reset_24h_accumulations`` places a sub-hourly step in its 24h window.

    The patch is for archives that wrongly encode ``startStep == endStep``;
    with minute-unit steps it must floor to the 24-hour boundary without
    losing the minutes.
    """
    valid = BASE + datetime.timedelta(hours=25, minutes=10)
    field = _GribField(BASE, "25h10m", "25h10m", valid)
    interval = FieldToInterval(patches=["reset_24h_accumulations"])(field)
    assert interval == SignedInterval(
        start=BASE + datetime.timedelta(hours=24),
        end=valid,
        base=BASE,
    )


def test_set_start_step_to_zero_patch_sub_hourly():
    field = _GribField(BASE, "10m", "30m", BASE + _minutes(30))
    interval = FieldToInterval(patches=["set_start_step_to_zero"])(field)
    assert interval == SignedInterval(start=BASE, end=BASE + _minutes(30), base=BASE)


def test_grib1_minute_unit_step_is_refused_not_mis_read():
    """A GRIB1 minute-unit step disagrees with the validity time: refuse loudly."""
    field = _Grib1MinuteField(BASE, 10, 10, BASE + _minutes(10))
    with pytest.raises(ValueError, match="cannot represent a sub-hourly step"):
        FieldToInterval()(field)


def test_grib1_whole_hour_step_in_minute_units_is_accepted():
    """The guard only fires on a genuine disagreement with the validity time.

    A step of 60 minutes is representable in hours, so eccodes returns ``1``
    and everything lines up.
    """
    field = _Grib1MinuteField(BASE, 0, 1, BASE + datetime.timedelta(hours=1))
    interval = FieldToInterval()(field)
    assert interval == SignedInterval(start=BASE, end=BASE + datetime.timedelta(hours=1), base=BASE)
