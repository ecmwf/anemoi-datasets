# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the factorised archive descriptions of the accumulate source.

Covers the ``accumulated:`` scheme grammar, the recurring ``base_dates``
form (with wildcard sugar), the exact candidate generation of
``TrajectoryIntervalGenerator``, and the ``AccumulateSchema`` recipe
validation rules.
"""

import datetime

import pytest
from pydantic import ValidationError

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate.description import AccumulateSchema
from anemoi.datasets.create.sources.accumulate.description import FromTrajectories
from anemoi.datasets.create.sources.accumulate.description import RecurringBaseDates
from anemoi.datasets.create.sources.accumulate.description import TrajectoryIntervalGenerator
from anemoi.datasets.create.sources.accumulate.description import infer_from_trajectories
from anemoi.datasets.create.sources.accumulate.description import parse_accumulated
from anemoi.datasets.create.sources.accumulate.interval_generators import increments_generator


def _dt(*args):
    return datetime.datetime(*args)


def _hours(n):
    return datetime.timedelta(hours=n)


# ---------------------------------------------------------------------------
# accumulated: grammar
# ---------------------------------------------------------------------------


def test_parse_accumulated_values():
    assert parse_accumulated("from-zero") == ("from-zero", None)
    assert parse_accumulated("from-previous-step") == ("from-previous-step", None)
    assert parse_accumulated("from-zero-reset-every-24h") == ("from-zero-reset", 24)
    assert parse_accumulated("from-zero-reset-every-1d") == ("from-zero-reset", 24)
    assert parse_accumulated("from-zero-reset-every-6h") == ("from-zero-reset", 6)


@pytest.mark.parametrize("bad", ["auto", "from-start", "from-zero-reset-every-", "from-zero-reset-every-30m", ""])
def test_parse_accumulated_rejects(bad):
    with pytest.raises(ValueError):
        parse_accumulated(bad)


# ---------------------------------------------------------------------------
# base_dates: recurring form and wildcard sugar
# ---------------------------------------------------------------------------


def test_base_dates_times_forms():
    for times in ([6, 18], ["06:00", "18:00"], ["6", "18"], [600, 1800]):
        bd = RecurringBaseDates.model_validate({"times": times})
        assert bd.times == [datetime.time(6), datetime.time(18)]


def test_base_dates_duplicate_times_rejected():
    with pytest.raises(ValidationError, match="duplicates"):
        RecurringBaseDates.model_validate({"times": [6, "06:00"]})


def test_base_dates_yaml_sexagesimal_footgun():
    # YAML parses an unquoted 18:00 as the integer 1080; the error says to quote it.
    with pytest.raises(ValidationError, match="quote"):
        RecurringBaseDates.model_validate({"times": [1080]})


def test_base_dates_wildcard_sugar():
    bd = RecurringBaseDates.model_validate("????-??-?? 06:00")
    assert bd.times == [datetime.time(6)]
    assert bd.day_of_month is None

    bd = RecurringBaseDates.model_validate(["????-??-?? 06:00", "????-??-?? 18:00"])
    assert bd.times == [datetime.time(6), datetime.time(18)]

    bd = RecurringBaseDates.model_validate("????-??-01 00:00")
    assert bd.times == [datetime.time(0)]
    assert bd.day_of_month == [1]


@pytest.mark.parametrize(
    "bad",
    [
        "????-??-?? ??:00",  # wildcard time
        "2020-??-?? 00:00",  # concrete year
        "????-01-?? 00:00",  # concrete month
        "????-??-0? 00:00",  # partial-wildcard day
        "????-??-??",  # no time part
    ],
)
def test_base_dates_wildcard_rejects(bad):
    with pytest.raises(ValidationError):
        RecurringBaseDates.model_validate(bad)


def test_base_dates_matches():
    bd = RecurringBaseDates.model_validate(
        {"times": [6, 18], "day_of_week": ["mon", "thursday"], "start": "2020-01-01", "end": "2020-12-31"}
    )
    assert bd.day_of_week == ["mon", "thu"]
    assert bd.matches(_dt(2020, 1, 6, 6))  # a Monday
    assert bd.matches(_dt(2020, 1, 9, 18))  # a Thursday
    assert not bd.matches(_dt(2020, 1, 7, 6))  # a Tuesday
    assert not bd.matches(_dt(2020, 1, 6, 12))  # wrong time
    assert not bd.matches(_dt(2019, 12, 30, 6))  # before start
    assert not bd.matches(_dt(2021, 1, 4, 6))  # after end


# ---------------------------------------------------------------------------
# step pairs implied by the description
# ---------------------------------------------------------------------------


def _description(**kwargs):
    payload = {
        "base_dates": {"times": [0]},
        "steps": {"start": "0h", "end": "6h", "frequency": "1h"},
        "accumulated": "from-previous-step",
    }
    payload.update(kwargs)
    return FromTrajectories.model_validate(payload)


def test_step_pairs_from_previous_step():
    d = _description()
    assert d.step_pairs() == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


def test_step_pairs_from_zero():
    d = _description(steps={"start": "1h", "end": "4h", "frequency": "1h"}, accumulated="from-zero")
    assert d.step_pairs() == [(0, 1), (0, 2), (0, 3), (0, 4)]


def test_step_pairs_reset():
    d = _description(
        steps={"start": "0h", "end": "50h", "frequency": "1h"},
        accumulated="from-zero-reset-every-24h",
    )
    pairs = {b: a for a, b in d.step_pairs()}  # end -> start
    assert pairs[1] == 0
    assert pairs[23] == 0
    assert pairs[24] == 0
    assert pairs[25] == 24
    assert pairs[48] == 24
    assert pairs[49] == 48
    assert pairs[50] == 48


def test_step_pairs_list_of_ranges():
    # rr se-al-ec: 1..6 by 1h then 6..30 by 3h
    d = _description(
        steps=[
            {"start": "1h", "end": "6h", "frequency": "1h"},
            {"start": "6h", "end": "30h", "frequency": "3h"},
        ],
        accumulated="from-zero",
    )
    assert d.step_grid_hours == [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30]


def test_from_previous_step_needs_two_steps():
    with pytest.raises(ValidationError, match="at least two steps"):
        _description(steps={"start": "6h", "end": "6h", "frequency": "6h"})


# ---------------------------------------------------------------------------
# TrajectoryIntervalGenerator: exact candidates + coverings
# ---------------------------------------------------------------------------


def test_candidates_era5_like():
    d = _description(
        base_dates={"times": [6, 18]},
        steps={"start": "0h", "end": "18h", "frequency": "1h"},
        accumulated="from-previous-step",
    )
    gen = TrajectoryIntervalGenerator(d)

    candidates = gen(_dt(2024, 1, 1, 19))
    # forward 19->20 from the 18Z run (step 1-2), forward 19->20 from the 06Z run (13-14),
    # backward 19->18 from both runs
    assert SignedInterval(_dt(2024, 1, 1, 19), _dt(2024, 1, 1, 20), base=_dt(2024, 1, 1, 18)) in candidates
    assert SignedInterval(_dt(2024, 1, 1, 19), _dt(2024, 1, 1, 20), base=_dt(2024, 1, 1, 6)) in candidates
    # most recent base first
    bases = [c.base for c in candidates]
    assert bases == sorted(bases, reverse=True)
    for c in candidates:
        assert c.start == _dt(2024, 1, 1, 19)


def test_covering_monthly_reset_far_from_base():
    """A day-15 window over a monthly archive — out of reach of the old ±1-day search."""
    d = _description(
        base_dates={"times": [0], "day_of_month": 1},
        steps={"start": "0h", "end": "720h", "frequency": "1h"},
        accumulated="from-zero-reset-every-24h",
    )
    gen = TrajectoryIntervalGenerator(d)
    base = _dt(2014, 6, 1, 0)

    cover = gen.covering_intervals(_dt(2014, 6, 15, 12), _dt(2014, 6, 15, 13))
    assert sum(i.length for i in cover) == _hours(1).total_seconds()
    assert all(i.base == base for i in cover)
    # -a(336, 348) + a(336, 349)
    assert set(cover) == {
        -SignedInterval(base + _hours(336), base + _hours(348), base=base),
        SignedInterval(base + _hours(336), base + _hours(349), base=base),
    }


def test_covering_bounded_archive():
    d = _description(
        base_dates={"times": [0, 12], "start": "2021-01-01", "end": "2021-01-31"},
        steps={"start": "1h", "end": "90h", "frequency": "1h"},
        accumulated="from-zero",
    )
    gen = TrajectoryIntervalGenerator(d)
    # inside the bounds: fine
    cover = gen.covering_intervals(_dt(2021, 1, 10, 18), _dt(2021, 1, 11, 0))
    assert sum(i.length for i in cover) == _hours(6).total_seconds()
    # outside the bounds: no coverage
    with pytest.raises(ValueError, match="Cannot find coverage"):
        gen.covering_intervals(_dt(2022, 6, 10, 18), _dt(2022, 6, 11, 0))


def test_infer_from_trajectories_auto():
    d = infer_from_trajectories("mars", {"class": "ea", "stream": "oper"})
    assert d.base_dates.times == [datetime.time(6), datetime.time(18)]
    assert d.accumulated == "from-previous-step"
    assert d.step_grid_hours == list(range(19))

    with pytest.raises(ValueError, match="only supported for the 'mars' source"):
        infer_from_trajectories("grib-index", {"index-db": "x"})

    with pytest.raises(ValueError, match="no 'class'"):
        infer_from_trajectories("mars", {"param": ["tp"]})


def test_increments_generator_spacing():
    gen = increments_generator(_hours(3))
    candidates = gen(_dt(2024, 1, 2, 9))
    forward = [c for c in candidates if c.length > 0]
    assert SignedInterval(_dt(2024, 1, 2, 9), _dt(2024, 1, 2, 12), base=None) in forward
    # increments are 3h-aligned to midnight: no 1h candidates
    assert all(abs(c.length) == _hours(3).total_seconds() for c in candidates)

    with pytest.raises(ValueError, match="divide 24h"):
        increments_generator(_hours(5))


# ---------------------------------------------------------------------------
# AccumulateSchema: recipe validation rules
# ---------------------------------------------------------------------------


def _schema(**kwargs):
    payload = {"period": "6h", "source": {"mars": {"class": "ea"}}}
    payload.update(kwargs)
    return AccumulateSchema.model_validate(payload)


def test_schema_accepts_each_description_key():
    s = _schema(**{"from-trajectories": "auto"})
    assert s.description_key == "from-trajectories"

    s = _schema(
        source={"grib-index": {"index-db": "x"}},
        **{"from-increments": "1h"},
    )
    assert s.description_key == "from-increments"
    assert s.from_increments == _hours(1)

    s = _schema(
        source={"grib-index": {"index-db": "x"}},
        **{"from-lookup-table": {"start": "1970-01-01", "0-6": [18, "6-12"]}},
    )
    assert s.description_key == "from-lookup-table"


def test_schema_exactly_one_description():
    with pytest.raises(ValidationError, match="only one archive description"):
        _schema(**{"from-trajectories": "auto", "from-increments": "1h"})

    with pytest.raises(ValidationError, match="describe the archive"):
        _schema()


def test_schema_description_excludes_block_accumulated():
    with pytest.raises(ValidationError, match="mutually"):
        _schema(**{"from-trajectories": "auto", "accumulated": "from-zero"})


def test_schema_deprecated_spellings():
    with pytest.deprecated_call():
        s = _schema(availability="auto")
    assert s.covering == {"auto": "auto"}
    assert s.availability is None

    with pytest.deprecated_call():
        s = _schema(covering={"auto": [(0, "0-6")]})
    assert s.description_key is None
    assert s.covering is not None

    with pytest.deprecated_call():
        s = _schema(accumulation="from-zero")
    assert s.accumulated == "from-zero"
    assert s.accumulation is None


def test_schema_unknown_key_rejected():
    with pytest.raises(ValidationError):
        _schema(coverings={"auto": "auto"})


def test_schema_unknown_patch_rejected():
    with pytest.raises(ValidationError, match="unknown patch"):
        _schema(**{"from-trajectories": "auto", "patch": ["no_such_patch"]})


def test_schema_known_patch_accepted():
    s = _schema(**{"from-trajectories": "auto", "patch": ["reset_24h_accumulations"]})
    assert s.patch == ["reset_24h_accumulations"]
