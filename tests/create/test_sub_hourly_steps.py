# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Sub-hourly steps through the recipe schema and the request encoders.

Request construction and schema validation only — no MARS or FDB access.
The complementary tests live in ``test_step_syntax.py`` (the boundary
helpers), ``test_field_to_interval_units.py`` (reading a field's own window)
and ``test_forecast_covering.py`` (the trajectory covering).
"""

import datetime

import pytest
from pydantic import ValidationError

import anemoi.datasets.create.sources.mars.source as mars_source_module
from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.sources.mars.source import MarsSource
from anemoi.datasets.create.time_schemas import Steps


def _minutes(n):
    return datetime.timedelta(minutes=n)


# ---------------------------------------------------------------------------
# The recipe schema
# ---------------------------------------------------------------------------


def test_steps_accept_sub_hourly_frequency():
    """``steps:`` is minute-resolution (it used to reject anything sub-hourly)."""
    steps = Steps(start="0h", end="1h", frequency="10m")
    assert len(steps) == 7
    assert list(steps)[1] == datetime.timedelta(minutes=10)


def test_steps_accept_sub_hourly_start_and_end():
    steps = Steps(start="10m", end="30m", frequency="10m")
    assert len(steps) == 3


def test_steps_reject_sub_minute_frequency():
    """Minute is the finest resolution the pipeline can express."""
    with pytest.raises(ValidationError, match="whole number of minutes"):
        Steps(start="0h", end="1h", frequency="30s")


def test_steps_reject_sub_minute_endpoints():
    """The endpoints are checked too (the span here is a whole 6 x 10 min)."""
    with pytest.raises(ValidationError, match="whole number of minutes"):
        Steps(start="30s", end="3630s", frequency="10m")


def test_steps_still_require_a_dividing_frequency():
    with pytest.raises(ValidationError, match="must divide"):
        Steps(start="0m", end="25m", frequency="10m")


# ---------------------------------------------------------------------------
# The MARS request encoder
# ---------------------------------------------------------------------------


class _Context:
    def trace(self, *args, **kwargs):
        pass


@pytest.fixture
def captured_requests(monkeypatch):
    captured = []

    def capture(context, requests, use_cdsapi_dataset=None):
        captured.extend(requests)
        return requests

    monkeypatch.setattr(mars_source_module, "fire_prebuilt_requests", capture)
    return captured


def _source(**extra):
    return MarsSource(
        _Context(),
        **{"class": "od", "stream": "oper", "type": "fc", "levtype": "sfc", "param": ["tp"], **extra},
    )


def test_forecast_dates_encode_sub_hourly_steps(captured_requests):
    """A sub-hourly lead time is sent in the archive's minute syntax."""
    bt = datetime.datetime(2024, 1, 1, 0)
    source = _source()
    source.execute_forecast_dates(
        ForecastDates([(bt + _minutes(10), bt), (bt + _minutes(70), bt), (bt + _minutes(120), bt)])
    )

    assert [r["step"] for r in captured_requests] == ["10m", "1h10m", 2]
    assert {r["date"] for r in captured_requests} == {"20240101"}
    assert {r["time"] for r in captured_requests} == {"0000"}


def test_forecast_dates_keep_whole_hour_steps_as_ints(captured_requests):
    """Rule: an hour-based request must be byte-for-byte what it always was."""
    bt = datetime.datetime(2024, 1, 1, 0)
    source = _source()
    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=6), bt)]))

    (request,) = captured_requests
    assert request["step"] == 6
    assert isinstance(request["step"], int)


def test_forecast_dates_reject_sub_minute_steps():
    bt = datetime.datetime(2024, 1, 1, 0)
    source = _source()
    with pytest.raises(ValueError, match="whole-minute steps"):
        source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(seconds=30), bt)]))


def test_valid_dates_expand_a_sub_hourly_step_list(monkeypatch):
    """``step: [10m, 20m]`` on the valid-dates side derives the right base time.

    The base is ``valid_date − step``; reading the step as whole hours used to
    raise ``ValueError`` before it got that far.
    """
    from anemoi.datasets.create.sources.mars import retrieval as retrieval_module

    captured = []

    def capture(context, requests, use_cdsapi_dataset=None):
        captured.extend(requests)
        return requests

    monkeypatch.setattr(retrieval_module, "fire_requests", capture, raising=False)
    monkeypatch.setattr(retrieval_module, "fire_prebuilt_requests", capture, raising=False)

    requests = retrieval_module._expand_mars_request(
        {"class": "od", "param": "tp", "step": ["10m", "20m"]},
        datetime.datetime(2024, 1, 1, 1, 0),
    )

    assert [(r["date"], r["time"], r["step"]) for r in requests] == [
        ("20240101", "0050", "10m"),
        ("20240101", "0040", "20m"),
    ]


def test_valid_dates_hourly_expansion_is_unchanged():
    from anemoi.datasets.create.sources.mars import retrieval as retrieval_module

    requests = retrieval_module._expand_mars_request(
        {"class": "od", "param": "tp", "step": [0, 6]},
        datetime.datetime(2024, 1, 1, 12, 0),
    )

    assert [(r["date"], r["time"], r["step"]) for r in requests] == [
        ("20240101", "1200", 0),
        ("20240101", "0600", 6),
    ]


# ---------------------------------------------------------------------------
# forcings, which clones fields onto (basetime, step) pairs
# ---------------------------------------------------------------------------


def test_forcings_step_is_not_floored_to_the_hour(monkeypatch):
    """A sub-hourly step used to be floored, collapsing several points onto step 0."""
    import anemoi.datasets.create.sources.forcings as forcings_module
    from anemoi.datasets.create.sources.forcings import ForcingsSource

    bt = datetime.datetime(2024, 1, 1, 0)
    pairs = [(bt + _minutes(10), bt), (bt + _minutes(20), bt), (bt + datetime.timedelta(hours=1), bt)]

    class _Field:
        def __init__(self, vt):
            self.time = type("T", (), {"valid_datetime": staticmethod(lambda vt=vt: vt)})()

        def get(self, key, default=None):
            return {"parameter.variable": "cos_latitude"}.get(key, default)

    class _FieldList:
        @staticmethod
        def from_source(name, source_or_dataset=None, date=None, param=None):
            return [_Field(vt) for vt in date]

        @staticmethod
        def from_fields(fields):
            return fields

    recorded = []

    class _FieldFactory:
        @staticmethod
        def with_new_metadata(field, **kwargs):
            recorded.append(kwargs)
            return field

    monkeypatch.setattr(forcings_module, "FieldList", _FieldList)
    monkeypatch.setattr(forcings_module, "Field", _FieldFactory)

    source = ForcingsSource(_Context(), template=None, param=["cos_latitude"])
    source.execute_forecast_dates(ForecastDates(pairs))

    assert [r["step"] for r in recorded] == [_minutes(10), _minutes(20), datetime.timedelta(hours=1)]
    assert {r["base_datetime"] for r in recorded} == {bt}
