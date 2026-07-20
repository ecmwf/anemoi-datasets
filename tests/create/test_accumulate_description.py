# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the factorised archive descriptions of the accumulate source.

Covers the ``from.accumulation:`` scheme grammar, the recurring ``base_dates``
form (with wildcard sugar), the exact candidate generation of
``TrajectoryIntervalGenerator``, and the ``AccumulateSchema`` recipe
validation rules.
"""

import datetime
import warnings

import pytest
from pydantic import ValidationError

from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.accumulate.description import AccumulateSchema
from anemoi.datasets.create.sources.accumulate.description import FromBare
from anemoi.datasets.create.sources.accumulate.description import FromLookupTable
from anemoi.datasets.create.sources.accumulate.description import FromTrajectories
from anemoi.datasets.create.sources.accumulate.description import RecurringBaseDates
from anemoi.datasets.create.sources.accumulate.description import TrajectoryIntervalGenerator
from anemoi.datasets.create.sources.accumulate.description import check_valid_time_source
from anemoi.datasets.create.sources.accumulate.description import infer_from_trajectories
from anemoi.datasets.create.sources.accumulate.description import parse_accumulation


def _dt(*args):
    return datetime.datetime(*args)


def _hours(n):
    return datetime.timedelta(hours=n)


# ---------------------------------------------------------------------------
# from.accumulation: grammar
# ---------------------------------------------------------------------------


def test_parse_accumulation_values():
    assert parse_accumulation("from-zero") == ("from-zero", None)
    assert parse_accumulation("1h") == ("increment", 1)
    assert parse_accumulation("3h") == ("increment", 3)
    assert parse_accumulation("from-zero-reset-every-24h") == ("from-zero-reset", 24)
    assert parse_accumulation("from-zero-reset-every-1d") == ("from-zero-reset", 24)
    assert parse_accumulation("from-zero-reset-every-6h") == ("from-zero-reset", 6)


@pytest.mark.parametrize("bad", ["auto", "from-start", "from-zero-reset-every-", "from-zero-reset-every-30m", ""])
def test_parse_accumulation_rejects(bad):
    with pytest.raises(ValueError):
        parse_accumulation(bad)


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


def test_base_dates_day_of_week_is_strict():
    """Exact 3-letter abbreviation or exact full name; no prefix matching."""
    bd = RecurringBaseDates.model_validate({"times": [0], "day_of_week": ["mon", "thursday"]})
    assert bd.day_of_week == ["mon", "thu"]
    for bad in ("mondi", "m", "tues"):
        with pytest.raises(ValidationError, match="day_of_week"):
            RecurringBaseDates.model_validate({"times": [0], "day_of_week": [bad]})


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
        "steps": {"start": "1h", "end": "6h", "frequency": "1h"},
        "accumulation": "1h",
    }
    payload.update(kwargs)
    return FromTrajectories.model_validate(payload)


def test_step_pairs_increment():
    """`steps` lists the fields; each covers (step - frequency, step)."""
    d = _description()
    assert d.step_pairs() == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    # a 3-hourly archive: fields at 3, 6, ... each covering 3h
    d = _description(steps={"start": "3h", "end": "12h", "frequency": "3h"}, accumulation="3h")
    assert d.step_pairs() == [(0, 3), (3, 6), (6, 9), (9, 12)]


def test_step_pairs_from_zero():
    d = _description(steps={"start": "1h", "end": "4h", "frequency": "1h"}, accumulation="from-zero")
    assert d.step_pairs() == [(0, 1), (0, 2), (0, 3), (0, 4)]


def test_step_pairs_reset():
    d = _description(
        steps={"start": "1h", "end": "50h", "frequency": "1h"},
        accumulation="from-zero-reset-every-24h",
    )
    pairs = {b: a for a, b in d.step_pairs()}  # end -> start
    assert pairs[1] == 0
    assert pairs[23] == 0
    assert pairs[24] == 0
    assert pairs[25] == 24
    assert pairs[48] == 24
    assert pairs[49] == 48
    assert pairs[50] == 48


def test_step_pairs_explicit_pairs():
    # An explicit "sA-sE" pair list is the whole description; no accumulation.
    # rr se-al-ec from-zero: 1..6 by 1h then 6..30 by 3h.
    d = _description(
        steps=[f"0-{s}" for s in (1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30)],
        accumulation=None,
    )
    assert d.is_explicit_pairs
    assert d.step_pairs() == [(0, s) for s in (1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30)]


def test_step_pairs_explicit_mixed_increments():
    """Explicit pairs express what a single scheme cannot: mixed accumulation lengths."""
    d = _description(steps=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-9", "9-12"], accumulation=None)
    assert d.step_pairs() == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 9), (9, 12)]


def test_from_layout_sentinel():
    """``from-layout`` inherits the run grid from the output layout."""
    d = _description(base_dates="from-layout", steps="from-layout", accumulation="from-zero")
    assert d.is_layout_grid
    assert not d.is_explicit_pairs
    # The grid comes from the layout at runtime, so grid queries are undefined.
    with pytest.raises(ValueError, match="not defined for a 'from-layout'"):
        d.step_pairs()
    with pytest.raises(ValueError, match="not defined for a 'from-layout'"):
        _ = d.step_grid_hours

    # A duration or a reset scheme are equally valid layout schemes.
    assert _description(base_dates="from-layout", steps="from-layout", accumulation="3h").is_layout_grid
    assert _description(
        base_dates="from-layout", steps="from-layout", accumulation="from-zero-reset-every-24h"
    ).is_layout_grid


def test_from_layout_must_be_on_both_keys():
    """The sentinel is paired: it must sit on both ``base_dates`` and ``steps``."""
    with pytest.raises(ValidationError, match="on both 'base_dates' and 'steps'"):
        _description(base_dates="from-layout", accumulation="from-zero")
    with pytest.raises(ValidationError, match="on both 'base_dates' and 'steps'"):
        _description(steps="from-layout", accumulation="from-zero")


def test_from_layout_requires_accumulation():
    """The layout supplies the grid, but the accumulation scheme must be stated."""
    with pytest.raises(ValidationError, match="needs an 'accumulation'"):
        _description(base_dates="from-layout", steps="from-layout", accumulation=None)


def test_step_pairs_accept_both_spellings():
    """A pair may be written ``"sA-sE"`` or ``[sA, sE]``, even mixed in one list."""
    want = [(0, 6), (6, 9)]
    assert _description(steps=["0-6", "6-9"], accumulation=None).step_pairs() == want
    assert _description(steps=[[0, 6], [6, 9]], accumulation=None).step_pairs() == want
    assert _description(steps=["0-6", [6, 9]], accumulation=None).step_pairs() == want


def test_partial_description_rejected_both_ways():
    """`base_dates` and `steps` are both required for a trajectory source."""
    with pytest.raises(ValidationError):
        _description(steps=None)
    with pytest.raises(ValidationError):
        _description(base_dates=None)


@pytest.mark.parametrize("accumulation", ["from-zero", "from-zero-reset-every-24h"])
def test_step_zero_rejected_under_every_scheme(accumulation):
    """No field exists at step 0 — a(0, 0) holds nothing — under any scheme."""
    with pytest.raises(ValidationError, match="no field\\s+exists at step 0"):
        _description(steps={"start": "0h", "end": "6h", "frequency": "1h"}, accumulation=accumulation)


def test_increment_rejects_start_below_accumulation():
    """The first field cannot start before the forecast does."""
    with pytest.raises(ValidationError, match="shorter than the 'accumulation' length"):
        _description(steps={"start": "1h", "end": "7h", "frequency": "3h"}, accumulation="3h")


def test_accumulation_length_is_independent_of_frequency():
    """A duration is the window length, decoupled from the step spacing."""
    # overlapping / rolling: 24 h windows archived every 6 h
    d = _description(steps={"start": "24h", "end": "48h", "frequency": "6h"}, accumulation="24h")
    assert d.step_pairs() == [(0, 24), (6, 30), (12, 36), (18, 42), (24, 48)]
    # sparse: 1 h windows archived every 6 h (gaps between them)
    d = _description(steps={"start": "6h", "end": "18h", "frequency": "6h"}, accumulation="1h")
    assert d.step_pairs() == [(5, 6), (11, 12), (17, 18)]


def test_list_of_range_dicts_is_rejected():
    """The old ``list[Steps]`` spelling is gone: a range is a single dict, and an
    irregular grid is an explicit ``"sA-sE"`` pair list.
    """
    with pytest.raises(ValidationError):
        _description(
            steps=[{"start": "1h", "end": "6h", "frequency": "1h"}, {"start": "6h", "end": "30h", "frequency": "3h"}],
            accumulation="from-zero",
        )


def test_pairs_forbid_accumulation():
    """An explicit pair list is the full description; a scheme would contradict it."""
    with pytest.raises(ValidationError, match="remove 'accumulation'"):
        _description(steps=["0-1", "1-2"], accumulation="from-zero")


def test_range_requires_accumulation():
    """A range says which end-steps exist but not how they accumulate."""
    with pytest.raises(ValidationError, match="needs an 'accumulation'"):
        _description(steps={"start": "1h", "end": "6h", "frequency": "1h"}, accumulation=None)


# ---------------------------------------------------------------------------
# TrajectoryIntervalGenerator: exact candidates + coverings
# ---------------------------------------------------------------------------


def test_candidates_era5_like():
    d = _description(
        base_dates={"times": [6, 18]},
        steps={"start": "1h", "end": "18h", "frequency": "1h"},
        accumulation="1h",
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
        steps={"start": "1h", "end": "720h", "frequency": "1h"},
        accumulation="from-zero-reset-every-24h",
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
        accumulation="from-zero",
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
    assert d.accumulation == "1h"
    assert d.step_grid_hours == list(range(1, 19))  # fields at MARS steps 1..18
    assert d.step_pairs() == [(s - 1, s) for s in range(1, 19)]

    with pytest.raises(ValueError, match="only supported for the 'mars' source"):
        infer_from_trajectories("grib-index", {"index-db": "x"})

    with pytest.raises(ValueError, match="no 'class'"):
        infer_from_trajectories("mars", {"param": ["tp"]})


# ---------------------------------------------------------------------------
# AccumulateSchema: recipe validation rules
# ---------------------------------------------------------------------------


def _schema(**kwargs):
    payload = {"period": "6h", "source": {"mars": {"class": "ea"}}}
    payload.update(kwargs)
    return AccumulateSchema.model_validate(payload)


def test_schema_accepts_each_from_kind():
    # base-less valid-time source: a bare duration
    s = _schema(
        source={"grib-index": {"index-db": "x"}},
        **{"from": {"accumulation": "1h"}},
    )
    assert isinstance(s.from_, FromBare)
    assert s.from_.duration == _hours(1)

    # the lookup table, nested under `lookup-table`
    s = _schema(
        source={"grib-index": {"index-db": "x"}},
        **{"from": {"lookup-table": {"start": "1970-01-01", "0-6": [18, "6-12"]}}},
    )
    assert isinstance(s.from_, FromLookupTable)
    assert s.from_.entries()["0-6"] == [18, "6-12"]


def test_schema_bare_form():
    """A bare `from:` states only the scheme; the layout/context supplies the grid."""
    s = _schema(**{"from": {"accumulation": "from-zero"}})
    assert isinstance(s.from_, FromBare)
    assert s.from_.accumulation == "from-zero"


def test_schema_from_defaults_to_none():
    """`from:` is optional; omitting it leaves `from_` as None (recognise from the source)."""
    assert _schema().from_ is None
    assert _schema().from_kind is None

    # recognition only knows well-known MARS archives — anything else still fails loudly
    from anemoi.datasets.create.sources.accumulate.description import infer_from_trajectories

    with pytest.raises(ValueError, match="only supported for the 'mars' source"):
        infer_from_trajectories("grib-index", {"index-db": "x"})


def test_schema_dump_round_trips_through_normalise_from():
    """The validated schema is dumped by field name and re-normalised by the runtime source.

    An omitted `from:` stays None through the round-trip: it dumps under the
    field name ``from_`` and re-validates without tripping the "'from: auto' is
    not a value" rejection (which only guards the user-facing ``from`` alias).
    """
    from anemoi.datasets.create.sources.accumulate.description import normalise_from

    # (a) omitted from: -> None -> dumped -> re-normalised without error
    dumped = _schema().model_dump()
    assert dumped["from_"] is None
    assert normalise_from(from_=dumped["from_"]) == (None, None)

    # ... and the dump re-validates as a whole (this dump is what ends up in
    # dataset metadata); the field-name `from_` is not the `from` alias, so the
    # user-written-auto rejection does not fire.
    assert AccumulateSchema.model_validate(dumped).from_ is None

    # (b) an explicit description round-trips to the same description
    dumped = _schema(**{"from": {"accumulation": "1h"}}, source={"grib-index": {"index-db": "x"}}).model_dump()
    resolved, covering = normalise_from(from_=dumped["from_"])
    assert covering is None
    assert isinstance(resolved, FromBare) and resolved.duration == _hours(1)


def test_schema_unknown_from_key_rejected():
    # there is no `type:` discriminator any more — an unexpected key is rejected
    with pytest.raises(ValidationError):
        _schema(**{"from": {"type": "trajectories", "accumulation": "1h"}})

    with pytest.raises(ValidationError):
        _schema(**{"from": {"nonsense": 1}})


def test_validate_from_messages_on_raw_configs():
    """The runtime path takes raw dicts, so it re-validates them structurally."""
    from anemoi.datasets.create.sources.accumulate.description import _validate_from

    # a bare mapping is now valid (base-less / trajectory-layout scheme)
    assert _validate_from({"accumulation": "1h"}).accumulation == "1h"

    # an omitted `from:` is None (never reaches _validate_from), but a bare
    # string that is not a mapping is still caught here
    with pytest.raises(ValueError, match="must be a mapping"):
        _validate_from("trajectories")

    assert isinstance(_validate_from({"lookup-table": {"start": "1970-01-01", "0-6": [18, "6-12"]}}), FromLookupTable)


def test_valid_time_rules():
    """Description-level rules for a base-less valid-time ``from:``.

    Only the ``from:`` description is checked here — not the source. Whether the
    source can serve base-less intervals is decided by the argument type the
    caller builds, not pre-validated (a base-anchored source rejects the
    base-less request at dispatch).
    """
    # the source data's accumulation must divide the requested period
    with pytest.raises(ValueError, match="must divide the requested 'period'"):
        check_valid_time_source(FromBare(accumulation="4h"), period=_hours(6))

    # ... and from-zero/reset cannot be base-less: they need base_dates + steps
    with pytest.raises(ValueError, match="must be a fixed duration"):
        check_valid_time_source(FromBare(accumulation="from-zero"), period=_hours(6))

    # a duration that does not divide 24h is fine when it divides the period:
    # the window is tiled directly, so there is no 24h-grid constraint.
    check_valid_time_source(FromBare(accumulation="5h"), period=_hours(5))
    check_valid_time_source(FromBare(accumulation="5h"), period=_hours(10))


def test_schema_only_one_description():
    with pytest.raises(ValidationError, match="only one source-data description"):
        _schema(**{"from": {"accumulation": "from-zero"}, "accumulation": "from-zero"})

    with pytest.raises(ValidationError, match="only one source-data description"):
        _schema(**{"from": {"accumulation": "from-zero"}, "covering": {"auto": "auto"}})


def test_normalise_from_warn_flag_silences_deprecations():
    """The runtime source re-normalises with warn=False, so a deprecated
    spelling warns once — at recipe validation — not once per layer.
    """
    from anemoi.datasets.create.sources.accumulate.description import normalise_from

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalise_from(covering={"auto": "auto"}, warn=False)
        normalise_from(availability="auto", warn=False)
        normalise_from(accumulation="from-zero", warn=False)
    assert not caught, [str(w.message) for w in caught]


def test_schema_deprecated_block_level_scheme_key():
    """Block-level ``accumulation:`` becomes a bare ``from: {accumulation: ...}``."""
    with pytest.deprecated_call():
        s = _schema(accumulation="from-zero")
    assert isinstance(s.from_, FromBare)
    assert s.from_.accumulation == "from-zero"
    assert s.accumulation is None


def test_schema_deprecated_covering_spellings():
    with pytest.deprecated_call():
        s = _schema(availability="auto")
    assert s.covering == {"auto": "auto"}
    assert s.availability is None

    with pytest.deprecated_call():
        s = _schema(covering={"auto": [(0, "0-6")]})
    assert s.covering is not None
    assert s.from_ is None


def test_schema_unknown_key_rejected():
    with pytest.raises(ValidationError):
        _schema(coverings={"auto": "auto"})


def test_schema_unknown_patch_rejected():
    with pytest.raises(ValidationError, match="unknown patch"):
        _schema(patch=["no_such_patch"])


def test_schema_known_patch_accepted():
    s = _schema(patch=["reset_24h_accumulations"])
    assert s.patch == ["reset_24h_accumulations"]


def test_schema_rejects_explicit_auto():
    """Inference is the default, so `from: auto` is redundant and refused."""
    with pytest.raises(ValidationError, match="omit 'from:' entirely"):
        _schema(**{"from": "auto"})


# ---------------------------------------------------------------------------
# Recipe-level rules for the orthogonal `from:`/layout model
# ---------------------------------------------------------------------------

_TRAJ_SOURCE = {"mars": {"class": "od", "type": "fc", "param": ["tp"]}}
_GRIB_INDEX_SOURCE = {"grib-index": {"indexdb": "/tmp/x.db", "param": ["tp"]}}


def _recipe_with_accumulate(from_block, *, trajectories, source=_TRAJ_SOURCE, period="6h"):
    """Build a minimal recipe carrying one accumulate block, for validation."""
    from anemoi.datasets.create.recipe import Recipe

    block = {"period": period, "source": source}
    if from_block is not None:
        block["from"] = from_block
    recipe = {"input": {"accumulate": block}}
    if trajectories:
        recipe["base_dates"] = {"start": "2021-01-01", "end": "2021-01-03", "frequency": "12h"}
        recipe["steps"] = {"start": "6h", "end": "30h", "frequency": "3h"}
        recipe["output"] = {"layout": "trajectories"}
    else:
        recipe["dates"] = {"start": "2021-01-10", "end": "2021-01-12", "frequency": "6h"}
    return Recipe(**recipe)


def test_recipe_from_layout_accepted_in_trajectory():
    """``from-layout`` is the layout's own run — accepted under a trajectory layout."""
    _recipe_with_accumulate(
        {"base_dates": "from-layout", "steps": "from-layout", "accumulation": "from-zero"},
        trajectories=True,
    )


def test_recipe_from_layout_rejected_outside_trajectory():
    """The sentinel only means something when a layout imposes the grid."""
    with pytest.raises(ValidationError, match="only valid in a 'layout: trajectories'"):
        _recipe_with_accumulate(
            {"base_dates": "from-layout", "steps": "from-layout", "accumulation": "from-zero"},
            trajectories=False,
        )


def test_recipe_explicit_trajectory_source_accepted_in_trajectory():
    """`from:` describes the subsource — an explicit-grid archive is fine in either layout."""
    _recipe_with_accumulate(
        {
            "base_dates": {"times": ["00:00", "12:00"]},
            "steps": {"start": "6h", "end": "12h", "frequency": "6h"},
            "accumulation": "from-zero",
        },
        trajectories=True,
    )


def test_recipe_auto_accepted_in_trajectory():
    """`auto` infers a well-known archive; it must work in both layouts."""
    _recipe_with_accumulate(None, trajectories=True)


def test_recipe_bare_valid_time_source_in_trajectory():
    """A bare duration `from:` is a base-less valid-time subsource, relabelled per row."""
    _recipe_with_accumulate(
        {"accumulation": "3h"},
        trajectories=True,
        source=_GRIB_INDEX_SOURCE,
    )


def test_recipe_bare_from_zero_in_trajectory_rejected():
    """`from-zero` is a run scheme — bare, it needs an explicit or from-layout grid."""
    with pytest.raises(ValidationError, match="must be a fixed duration"):
        _recipe_with_accumulate({"accumulation": "from-zero"}, trajectories=True, source=_GRIB_INDEX_SOURCE)


def test_recipe_bare_valid_time_on_base_anchored_source_passes_validation():
    """Recipe validation does not pre-judge the source: a bare valid-time `from:`
    on a base-anchored source (mars) validates fine — the mismatch surfaces later
    at dispatch (the base-less request needs a basetime mars cannot supply).
    """
    _recipe_with_accumulate({"accumulation": "3h"}, trajectories=True, source=_TRAJ_SOURCE)
