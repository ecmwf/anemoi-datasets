# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the mars ``hindcast`` option (reforecast date/hdate requests).

Request construction only — no MARS access needed. ``fire_prebuilt_requests``
is stubbed out to capture the per-item request dicts.
"""

import datetime

import pytest

import anemoi.datasets.create.sources.mars.source as mars_source_module
from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.mars.source import MarsSource
from anemoi.datasets.create.sources.mars.source import _hindcast_refdate_mapping

# Mondays and Thursdays from 2023-06-29 to 2024-11-11 — the reforecast
# reference-date convention of the eefh/enfh downscaling datasets.
HINDCAST = {
    "reference_start": "2023-06-29",
    "reference_end": "2024-11-11",
    "day_of_week": ["monday", "thursday"],
    "years": 20,
}


class _Context:
    """Minimal context: MarsSource only needs ``trace`` besides the registry plumbing."""

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
        **{
            "class": "od",
            "stream": "eefh",
            "type": "cf",
            "levtype": "sfc",
            "param": ["2t"],
            "hindcast": HINDCAST,
            **extra,
        },
    )


# ---------------------------------------------------------------------------
# The hdate → refdate mapping
# ---------------------------------------------------------------------------


def test_mapping_maps_hdates_to_their_refdate() -> None:
    mapping = _hindcast_refdate_mapping(HINDCAST)

    # 2023-06-29 is a Thursday: its 20 hindcast years are 2003..2022.
    for year in range(2003, 2023):
        assert mapping[datetime.date(year, 6, 29)] == datetime.date(2023, 6, 29)

    # 2024-11-11 is a Monday: its hindcast years are 2004..2023.
    for year in range(2004, 2024):
        assert mapping[datetime.date(year, 11, 11)] == datetime.date(2024, 11, 11)

    # A month-day that matches no Monday/Thursday reference date is absent.
    # 2023-07-01 is a Saturday and 2024-07-01 is a Monday: present via 2024 only.
    assert mapping[datetime.date(2010, 7, 1)] == datetime.date(2024, 7, 1)


def test_mapping_size_is_years_times_refdates() -> None:
    mapping = _hindcast_refdate_mapping(HINDCAST)
    # Mon/Thu weekdays shift year-to-year, so no month-day is a reference
    # date in both 2023 and 2024: every (refdate, year) pair is distinct.
    n_refdates = len({r for r in mapping.values()})
    assert len(mapping) == 20 * n_refdates


def test_mapping_rejects_ambiguous_hdates() -> None:
    # Without a day_of_week restriction, a window longer than a year repeats
    # month-days: each hdate would map to two reference dates.
    with pytest.raises(ValueError, match="two reference dates"):
        _hindcast_refdate_mapping(
            {
                "reference_start": "2023-06-29",
                "reference_end": "2024-11-11",
            }
        )


def test_mapping_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="unknown option"):
        _hindcast_refdate_mapping({**HINDCAST, "typo_option": 1})


def test_day_of_week_accepts_abbreviations() -> None:
    """Same dialect as the accumulate `from.base_dates.day_of_week` selector."""
    abbreviated = _hindcast_refdate_mapping({**HINDCAST, "day_of_week": ["mon", "thu"]})
    assert abbreviated == _hindcast_refdate_mapping(HINDCAST)

    with pytest.raises(ValueError, match="invalid day_of_week"):
        _hindcast_refdate_mapping({**HINDCAST, "day_of_week": ["m"]})
    with pytest.raises(ValueError, match="invalid day_of_week"):
        _hindcast_refdate_mapping({**HINDCAST, "day_of_week": ["mondi"]})


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def test_forecast_dates_requests(captured_requests) -> None:
    source = _source()
    bt = datetime.datetime(2013, 6, 29)
    dates = ForecastDates([(bt + datetime.timedelta(hours=12), bt), (bt + datetime.timedelta(hours=24), bt)])

    source.execute_forecast_dates(dates)

    assert len(captured_requests) == 2
    for r, step in zip(captured_requests, (12, 24)):
        assert r["date"] == "20230629"
        assert r["hdate"] == "20130629"
        assert r["time"] == "0000"
        assert r["step"] == step
        assert r["stream"] == "eefh"


def test_forecast_intervals_requests(captured_requests) -> None:
    source = _source(param=["tp"])
    bt = datetime.datetime(2013, 6, 29)
    vt = bt + datetime.timedelta(hours=24)
    period = datetime.timedelta(hours=12)
    # from-zero covering of [vt-12h, vt]: +a(0,24) − a(0,12)
    intervals = [
        SignedInterval(start=bt, end=vt, base=bt),
        SignedInterval(start=vt - period, end=bt, base=bt),
    ]
    dates = ForecastIntervals(items=[(vt, bt, period)], intervals=intervals)

    source.execute_forecast_intervals(dates)

    assert len(captured_requests) == 2
    steps = sorted(r["step"] for r in captured_requests)
    assert steps == [12, 24]
    for r in captured_requests:
        assert r["date"] == "20230629"
        assert r["hdate"] == "20130629"
        assert r["time"] == "0000"


def test_without_hindcast_option_requests_are_unchanged(captured_requests) -> None:
    source = MarsSource(_Context(), **{"class": "od", "stream": "oper", "type": "fc", "param": ["2t"]})
    bt = datetime.datetime(2013, 6, 29, 12)
    dates = ForecastDates([(bt + datetime.timedelta(hours=6), bt)])

    source.execute_forecast_dates(dates)

    (r,) = captured_requests
    assert r["date"] == "20130629"
    assert r["time"] == "1200"
    assert r["step"] == 6
    assert "hdate" not in r


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_valid_dates_rejected() -> None:
    with pytest.raises(ValueError, match="forecast contexts"):
        _source().execute_valid_dates(ValidDates([datetime.datetime(2013, 6, 29)]))


def test_intervals_rejected() -> None:
    with pytest.raises(ValueError, match="forecast contexts"):
        _source().execute_intervals(Intervals([datetime.datetime(2013, 6, 29)], []))


def test_non_midnight_basetime_rejected() -> None:
    bt = datetime.datetime(2013, 6, 29, 12)
    with pytest.raises(ValueError, match="00Z"):
        _source().execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=6), bt)]))


def test_basetime_not_an_hdate_rejected() -> None:
    # 2013-07-01: 2023-07-01 is a Saturday, 2024-07-01 a Monday, so the hdate
    # exists for years 2004..2023 of 2024-07-01 — but 2013-07-02 (2023-07-02
    # Sunday, 2024-07-02 Tuesday) matches no Monday/Thursday reference date.
    bt = datetime.datetime(2013, 7, 2)
    with pytest.raises(ValueError, match="not a hindcast date"):
        _source().execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=6), bt)]))
