# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import pytest

from anemoi.datasets.create.recipe.dates import BaseDates
from anemoi.datasets.create.recipe.dates import StartEndDates


def _dt(s: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(s)


def test_wildcard_with_time_matches_across_years() -> None:
    """A ``????-MM-DD HH:MM`` pattern selects that instant in every year."""
    dates = StartEndDates(
        start="2020-01-01 00:00",
        end="2022-12-31 12:00",
        frequency="12h",
        missing=["????-01-02 00:00"],
    )
    assert dates.missing == [
        _dt("2020-01-02 00:00"),
        _dt("2021-01-02 00:00"),
        _dt("2022-01-02 00:00"),
    ]


def test_wildcard_date_only_matches_every_time_on_the_day() -> None:
    """A pattern with no time part matches all times on the selected days."""
    dates = StartEndDates(
        start="2021-01-01 00:00",
        end="2021-01-05 12:00",
        frequency="12h",
        missing=["????-01-02"],
    )
    assert dates.missing == [_dt("2021-01-02 00:00"), _dt("2021-01-02 12:00")]


def test_wildcard_mixes_with_ranges_and_explicit_dates() -> None:
    dates = StartEndDates(
        start="2020-12-30 00:00",
        end="2021-01-03 12:00",
        frequency="12h",
        missing=[
            {"start": "2020-12-30 12:00", "end": "2020-12-30 12:00"},
            "????-01-01 00:00",
        ],
    )
    assert dates.missing == [_dt("2020-12-30 12:00"), _dt("2021-01-01 00:00")]
    assert _dt("2020-12-30 12:00") not in dates.values
    assert _dt("2021-01-01 00:00") not in dates.values


def test_wildcard_is_inherited_by_base_dates() -> None:
    dates = BaseDates(
        start="2020-01-01 00:00",
        end="2022-12-31 12:00",
        frequency="12h",
        missing=["????-06-15 12:00"],
    )
    assert dates.missing == [
        _dt("2020-06-15 12:00"),
        _dt("2021-06-15 12:00"),
        _dt("2022-06-15 12:00"),
    ]
    # BaseDates keeps the slots for missing base dates in ``values``.
    assert _dt("2021-06-15 12:00") in dates.values


def test_wildcard_matching_nothing_warns_and_is_ignored(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        dates = StartEndDates(
            start="2021-01-01 00:00",
            end="2021-01-02 00:00",
            frequency="12h",
            missing=["????-07-01 00:00"],
        )
    assert dates.missing == []
    assert "matched no date" in caplog.text


@pytest.mark.parametrize("bad", ["??-01-02", "????/01/02", "????-1-02"])
def test_malformed_wildcard_matches_nothing(bad: str, caplog: pytest.LogCaptureFixture) -> None:
    # fnmatch is lenient: a pattern whose fixed parts do not line up simply
    # matches no date (and is reported), rather than raising.
    with caplog.at_level("WARNING"):
        dates = StartEndDates(start="2021-01-01 00:00", end="2021-01-02 00:00", frequency="12h", missing=[bad])
    assert dates.missing == []
    assert "matched no date" in caplog.text
