# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``anemoi-datasets recipe --migrate`` on accumulate blocks.

The migration rewrites the pre-redesign spellings (``accumulations`` source,
``availability:``, ``covering:``, ``accumulation:``) to the new, structural
``from:`` block (``base_dates``/``steps`` for trajectories, ``lookup-table``
for the table, a bare ``accumulation`` duration otherwise), and the result
must pass recipe validation.
"""

import datetime
import warnings

from anemoi.datasets.commands.recipe.migrate import migrate
from anemoi.datasets.commands.recipe.validate import validate_recipe


def _recipe(block: dict, trajectories: bool = False) -> dict:
    recipe = {"input": {"accumulate": block}}
    if trajectories:
        recipe["base_dates"] = {"start": "2021-01-01", "end": "2021-01-03", "frequency": "12h"}
        recipe["steps"] = {"start": "6h", "end": "30h", "frequency": "3h"}
        recipe["output"] = {"layout": "trajectories"}
    else:
        recipe["dates"] = {"start": "2021-01-10", "end": "2021-01-12", "frequency": "6h"}
    return recipe


def _block(recipe: dict) -> dict:
    return recipe["input"]["accumulate"]


def _migrate_and_validate(recipe: dict) -> dict:
    migrated = migrate(recipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validate_recipe(migrated)
    return migrated


MARS = {"mars": {"class": "ea", "type": "fc", "param": ["tp"]}}


def test_availability_auto_migrates_to_no_description():
    """`auto` is the default, so the migrated recipe simply carries no `from:`."""
    old = _recipe({"period": "6h", "availability": "auto", "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert "availability" not in block
    assert "covering" not in block
    assert "from" not in block


def test_covering_auto_auto_migrates_to_no_description():
    old = _recipe({"period": "6h", "covering": {"auto": "auto"}, "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert "from" not in block


def test_frequency_string_becomes_from_increments():
    old = _recipe({"period": "6h", "availability": "1h", "source": {"grib-index": {"index-db": "x", "param": ["tp"]}}})
    block = _block(_migrate_and_validate(old))
    assert block["from"] == {"accumulation": "1h"}


def test_explicit_from_zero_list_is_factorised():
    old = _recipe(
        {
            "period": "6h",
            "availability": [(0, "0-6/0-12"), (12, "0-6/0-12")],
            "source": {"mars": {"class": "od", "type": "fc", "param": ["tp"]}},
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from"] == {
        "base_dates": {"times": [0, 12]},
        "steps": {"start": "6h", "end": "12h", "frequency": "6h"},
        "accumulation": "from-zero",
    }


def test_explicit_increment_list_is_factorised():
    old = _recipe(
        {
            "period": "6h",
            "availability": [[0, "0-6/6-12/12-18/18-24"]],
            "source": MARS,
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from"] == {
        "base_dates": {"times": [0]},
        "steps": {"start": "6h", "end": "24h", "frequency": "6h"},
        "accumulation": "6h",
    }


def test_sugar_types_are_factorised():
    old = _recipe(
        {
            "period": "6h",
            "availability": {
                "type": "accumulated-from-previous-step",
                "basetime": [6, 18],
                "frequency": 3,
                "last_step": 18,
            },
            "source": MARS,
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from"] == {
        "base_dates": {"times": [6, 18]},
        "steps": {"start": "3h", "end": "18h", "frequency": "3h"},
        "accumulation": "3h",
    }

    old = _recipe(
        {
            "period": "6h",
            "availability": {
                "type": "accumulated-from-start",
                "basetime": [0, 12],
                "frequency": 6,
                "last_step": 18,
            },
            "source": {"mars": {"class": "od", "type": "fc", "param": ["tp"]}},
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from"] == {
        "base_dates": {"times": [0, 12]},
        "steps": {"start": "6h", "end": "18h", "frequency": "6h"},
        "accumulation": "from-zero",
    }


def test_reset_pattern_is_factorised():
    steps = []
    for start in range(0, 720, 24):
        for end in range(start + 1, start + 25):
            steps.append(f"{start}-{end}")
    old = _recipe(
        {
            "period": "1h",
            "availability": {
                "base_date": {"day_of_month": 1},
                "base_time": 0,
                "steps": "/".join(steps),
            },
            "patch": ["reset_24h_accumulations"],
            "source": {"mars": {"class": "rd", "type": "fc", "expver": "i6aj", "param": ["tp"]}},
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from"] == {
        "base_dates": {"times": [0], "day_of_month": 1},
        "steps": {"start": "1h", "end": "720h", "frequency": "1h"},
        "accumulation": "from-zero-reset-every-24h",
    }
    assert block["patch"] == ["reset_24h_accumulations"]


def test_irregular_grid_becomes_explicit_pairs():
    # rr se-al-ec-like: from-zero over 1..6 by 1 then 9..30 by 3. An irregular
    # grid has no range spelling, so migration emits explicit "sA-sE" pairs and
    # no 'accumulation' (the pairs are the whole description).
    pairs = [(0, i) for i in [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30]]
    old = _recipe(
        {
            "period": "3h",
            "availability": [[0, pairs]],
            "source": {"mars": {"class": "rr", "origin": "se-al-ec", "type": "fc", "param": ["tp"]}},
        }
    )
    block = _block(_migrate_and_validate(old))
    assert "accumulation" not in block["from"]
    assert block["from"]["steps"] == [f"0-{i}" for i in [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30]]


def test_mars_availability_dict_uses_known_archive_description():
    old = _recipe(
        {
            "period": "6h",
            "availability": {"mars": {"class": "ea", "stream": "oper"}},
            "source": MARS,
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from"]["accumulation"] == "1h"
    assert block["from"]["steps"] == {"start": "1h", "end": "18h", "frequency": "1h"}


def test_cycle_becomes_from_lookup_table():
    table = {
        "start": "1970-01-01",
        "0-6": [18, "6-12"],
        "6-12": [18, "12-18"],
        "12-18": [18, "18-24"],
        "18-24": [18, "0-6"],
    }
    old = _recipe(
        {
            "period": "6h",
            "covering": {"auto": {"cycle": dict(table)}},
            "source": {"grib-index": {"index-db": "x", "param": ["tp"]}},
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from"] == {"lookup-table": table}


def test_block_level_scheme_moves_into_from():
    """In a trajectory recipe the scheme key moves inside a ``from-layout`` ``from:``."""
    old = _recipe({"period": "1h", "accumulation": "from-zero", "source": MARS}, trajectories=True)
    block = _block(_migrate_and_validate(old))
    assert "accumulation" not in block
    assert block["from"] == {"base_dates": "from-layout", "steps": "from-layout", "accumulation": "from-zero"}


def test_covering_is_dropped_in_trajectory_recipe():
    """The old code silently ignored covering in the trajectory branch; migrate drops it."""
    old = _recipe(
        {"period": "1h", "accumulation": "from-zero", "covering": {"auto": "auto"}, "source": MARS},
        trajectories=True,
    )
    block = _block(_migrate_and_validate(old))
    assert "covering" not in block
    assert block["from"] == {"base_dates": "from-layout", "steps": "from-layout", "accumulation": "from-zero"}


def test_unfactorisable_description_is_left_unchanged():
    # different step lists per base time cannot be factorised
    weird = [(0, "0-6"), (12, "0-3/0-6")]
    old = _recipe({"period": "6h", "availability": weird, "source": MARS})
    migrated = migrate(old)
    block = _block(migrated)
    assert block["availability"] == weird
    assert "from" not in block


def test_migrate_is_idempotent_on_new_api():
    new = _recipe(
        {
            "period": "6h",
            "from": {
                "base_dates": {"times": [6, 18]},
                "steps": {"start": "1h", "end": "18h", "frequency": "1h"},
                "accumulation": "1h",
            },
            "source": MARS,
        }
    )
    assert migrate(new) == new


def test_old_accumulations_source_chains_to_new_api():
    """The pre-#326 'accumulations' source migrates all the way to the new API."""
    old = {
        "dates": {"start": "2021-01-10", "end": "2021-01-12", "frequency": "6h"},
        "input": {
            "join": [
                {
                    "accumulations": {
                        "class": "ea",
                        "expver": "0001",
                        "levtype": "sfc",
                        "param": ["tp"],
                        "accumulation_period": 6,
                    }
                }
            ]
        },
    }
    migrated = _migrate_and_validate(old)
    block = migrated["input"]["join"][0]["accumulate"]
    assert "availability" not in block
    assert "from" not in block
    assert block["period"] == 6
    assert block["source"]["mars"]["param"] == ["tp"]


def test_migrated_result_validates_without_deprecations():
    """The migrated recipe is fully on the new API: validating it must not warn."""
    old = _recipe({"period": "6h", "availability": "auto", "source": MARS})
    migrated = migrate(old)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_recipe(migrated)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, [str(w.message) for w in deprecations]


def test_increment_migrates_to_field_steps():
    """`steps` lists the fields, so a chained legacy pair set loses its leading 0."""
    old = _recipe({"period": "1h", "availability": [[6, "0-1/1-2/2-3"]], "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert block["from"]["accumulation"] == "1h"
    # fields exist at steps 1, 2, 3 — not at 0
    assert block["from"]["steps"] == {"start": "1h", "end": "3h", "frequency": "1h"}


def test_irregular_chained_pairs_are_not_factorised():
    """With a per-step duration the spacing IS the accumulation length, so an
    irregular chain has no valid spelling and is left alone.
    """
    old = _recipe({"period": "6h", "availability": [[0, "0-1/1-2/2-5"]], "source": MARS})
    block = _block(migrate(old))
    assert "from" not in block
    assert block["availability"] == [[0, "0-1/1-2/2-5"]]


def test_single_pair_with_nonzero_start_is_faithful():
    """A lone (basetime, "sA-sE") pair keeps its accumulation length.

    Regression: the frequency of an isolated step used to be guessed as the
    step value itself, turning a(6, 12) into a(0, 12) — a different field.
    """
    from anemoi.datasets.create.sources.accumulate.description import FromTrajectories

    old = _recipe({"period": "6h", "availability": [[6, "6-12"]], "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert block["from"]["steps"] == {"start": "12h", "end": "12h", "frequency": "6h"}
    d = FromTrajectories.model_validate(block["from"])
    assert d.step_pairs() == [(datetime.timedelta(hours=6), datetime.timedelta(hours=12))]


def test_block_level_scheme_is_dropped_outside_trajectory_recipes():
    """Outside trajectory recipes 'accumulation:' was never read; migrate drops it."""
    old = _recipe({"period": "6h", "accumulation": "from-zero", "covering": {"auto": "auto"}, "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert "accumulation" not in block
    assert "covering" not in block
    assert "from" not in block  # covering was auto = the default


# ---------------------------------------------------------------------------
# The safety net: a wrong or invalid migration is never produced — the block
# is left unchanged (deprecated spellings still run) with a warning.
# ---------------------------------------------------------------------------


def _assert_left_unchanged(block: dict) -> None:
    migrated = migrate(_recipe(dict(block)))
    assert _block(migrated) == block, _block(migrated)


def test_sugar_with_indivisible_last_step_is_left_unchanged():
    """Legacy last_step=15/frequency=6 generated fields beyond last_step; the
    factorised spelling would not validate, so the block is left alone.
    """
    _assert_left_unchanged(
        {
            "period": "6h",
            "availability": {"type": "accumulated-from-start", "basetime": [0], "frequency": 6, "last_step": 15},
            "source": MARS,
        }
    )


def test_wildcard_base_in_list_form_is_left_unchanged():
    """The legal legacy '*' base (any base time) has no factorised spelling."""
    _assert_left_unchanged({"period": "1h", "availability": [["*", "0-1/1-2/2-3"]], "source": MARS})


def test_frequency_string_indivisible_by_24h_migrates():
    """'5h' need not divide 24h: it divides the 10h period, so it migrates to a bare
    valid-time 'from:' (the window is tiled directly, not on a midnight-aligned grid).
    """
    old = _recipe({"period": "10h", "availability": "5h", "source": {"grib-index": {"indexdb": "x", "param": ["tp"]}}})
    block = _block(_migrate_and_validate(old))
    assert "availability" not in block
    assert block["from"] == {"accumulation": "5h"}


def test_frequency_string_migrates_regardless_of_source():
    """A frequency `availability:` becomes a bare valid-time `from:`; the source is not
    pre-judged (a base-anchored source rejecting it is a dispatch-time concern).
    """
    old = _recipe({"period": "6h", "availability": "1h", "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert "availability" not in block
    assert block["from"] == {"accumulation": "1h"}


def test_conflicting_descriptions_are_left_unchanged():
    """A block the schema would reject must stay rejectable, not silently resolved."""
    _assert_left_unchanged(
        {
            "period": "6h",
            "from": {"type": "trajectories", "accumulation": "from-zero"},
            "availability": [[12, "0-6/0-12"]],
            "source": MARS,
        }
    )
    _assert_left_unchanged({"period": "6h", "availability": "auto", "covering": {"auto": [[0, "0-6"]]}, "source": MARS})


def test_degenerate_pairs_are_left_unchanged():
    """'6-6' names no field and '6-0' is negative; factorising would invent data."""
    _assert_left_unchanged({"period": "6h", "availability": [[0, "6-6"]], "source": MARS})
    _assert_left_unchanged({"period": "6h", "availability": [[0, "6-0"]], "source": MARS})


def test_garbage_input_does_not_crash_migrate():
    _assert_left_unchanged(
        {
            "period": "6h",
            "availability": {
                "type": "accumulated-from-start",
                "basetime": ["2020-01-01 06:00"],
                "frequency": 6,
                "last_step": 12,
            },
            "source": MARS,
        }
    )


def test_duplicate_base_time_spellings_are_deduplicated():
    old = _recipe({"period": "6h", "availability": [[0, "0-6/0-12"], ["00:00", "0-6/0-12"]], "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert block["from"]["base_dates"] == {"times": [0]}


def test_unknown_base_date_selector_is_left_unchanged():
    """A base_date constraint other than day_of_month must not be dropped."""
    _assert_left_unchanged(
        {
            "period": "1h",
            "availability": {"base_date": {"day_of_week": "monday"}, "base_time": 0, "steps": "0-1/1-2"},
            "source": MARS,
        }
    )


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            obj()
            print(f"{name}: ok")
