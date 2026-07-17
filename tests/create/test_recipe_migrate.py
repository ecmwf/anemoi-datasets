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
``availability:``, ``covering:``, ``accumulation:``) to the new description
keys (``from-trajectories:`` / ``from-increments:`` / ``from-lookup-table:``
and ``accumulated:``), and the result must pass recipe validation.
"""

import warnings

import pytest

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


def test_availability_auto_becomes_from_trajectories_auto():
    old = _recipe({"period": "6h", "availability": "auto", "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert "availability" not in block
    assert "covering" not in block
    assert block["from-trajectories"] == "auto"


def test_covering_auto_auto_becomes_from_trajectories_auto():
    old = _recipe({"period": "6h", "covering": {"auto": "auto"}, "source": MARS})
    block = _block(_migrate_and_validate(old))
    assert block["from-trajectories"] == "auto"


def test_frequency_string_becomes_from_increments():
    old = _recipe(
        {"period": "6h", "availability": "1h", "source": {"grib-index": {"index-db": "x", "param": ["tp"]}}}
    )
    block = _block(_migrate_and_validate(old))
    assert block["from-increments"] == "1h"


def test_explicit_from_zero_list_is_factorised():
    old = _recipe(
        {
            "period": "6h",
            "availability": [(0, "0-6/0-12"), (12, "0-6/0-12")],
            "source": {"mars": {"class": "od", "type": "fc", "param": ["tp"]}},
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from-trajectories"] == {
        "base_dates": {"times": [0, 12]},
        "steps": {"start": "6h", "end": "12h", "frequency": "6h"},
        "accumulated": "from-zero",
    }


def test_explicit_from_previous_step_list_is_factorised():
    old = _recipe(
        {
            "period": "6h",
            "availability": [[0, "0-6/6-12/12-18/18-24"]],
            "source": MARS,
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from-trajectories"] == {
        "base_dates": {"times": [0]},
        "steps": {"start": "0h", "end": "24h", "frequency": "6h"},
        "accumulated": "from-previous-step",
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
    assert block["from-trajectories"] == {
        "base_dates": {"times": [6, 18]},
        "steps": {"start": "0h", "end": "18h", "frequency": "3h"},
        "accumulated": "from-previous-step",
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
    assert block["from-trajectories"] == {
        "base_dates": {"times": [0, 12]},
        "steps": {"start": "6h", "end": "18h", "frequency": "6h"},
        "accumulated": "from-zero",
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
    assert block["from-trajectories"] == {
        "base_dates": {"times": [0], "day_of_month": 1},
        "steps": {"start": "1h", "end": "720h", "frequency": "1h"},
        "accumulated": "from-zero-reset-every-24h",
    }
    assert block["patch"] == ["reset_24h_accumulations"]


def test_irregular_grid_becomes_list_of_ranges():
    # rr se-al-ec-like: from-zero over 1..6 by 1 then 9..30 by 3
    pairs = [(0, i) for i in [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27, 30]]
    old = _recipe(
        {
            "period": "3h",
            "availability": [[0, pairs]],
            "source": {"mars": {"class": "rr", "origin": "se-al-ec", "type": "fc", "param": ["tp"]}},
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from-trajectories"]["accumulated"] == "from-zero"
    assert block["from-trajectories"]["steps"] == [
        {"start": "1h", "end": "6h", "frequency": "1h"},
        {"start": "9h", "end": "30h", "frequency": "3h"},
    ]


def test_mars_availability_dict_uses_known_archive_description():
    old = _recipe(
        {
            "period": "6h",
            "availability": {"mars": {"class": "ea", "stream": "oper"}},
            "source": MARS,
        }
    )
    block = _block(_migrate_and_validate(old))
    assert block["from-trajectories"]["accumulated"] == "from-previous-step"
    assert block["from-trajectories"]["steps"] == {"start": "0h", "end": "18h", "frequency": "1h"}


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
    assert block["from-lookup-table"] == table


def test_accumulation_becomes_accumulated_in_trajectory_recipe():
    old = _recipe({"period": "1h", "accumulation": "from-zero", "source": MARS}, trajectories=True)
    block = _block(_migrate_and_validate(old))
    assert "accumulation" not in block
    assert block["accumulated"] == "from-zero"


def test_covering_is_dropped_in_trajectory_recipe():
    """The old code silently ignored covering in the trajectory branch; migrate drops it."""
    old = _recipe(
        {"period": "1h", "accumulation": "from-zero", "covering": {"auto": "auto"}, "source": MARS},
        trajectories=True,
    )
    block = _block(_migrate_and_validate(old))
    assert "covering" not in block
    assert "from-trajectories" not in block
    assert block["accumulated"] == "from-zero"


def test_unfactorisable_description_is_left_unchanged():
    # different step lists per base time cannot be factorised
    weird = [(0, "0-6"), (12, "0-3/0-6")]
    old = _recipe({"period": "6h", "availability": weird, "source": MARS})
    migrated = migrate(old)
    block = _block(migrated)
    assert block["availability"] == weird
    assert "from-trajectories" not in block


def test_migrate_is_idempotent_on_new_api():
    new = _recipe(
        {
            "period": "6h",
            "from-trajectories": {
                "base_dates": {"times": [6, 18]},
                "steps": {"start": "0h", "end": "18h", "frequency": "1h"},
                "accumulated": "from-previous-step",
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
    assert block["from-trajectories"] == "auto"
    assert block["period"] == 6
    assert block["source"]["mars"]["param"] == ["tp"]


def test_migrated_result_passes_validation_with_expected_deprecations():
    old = _recipe({"period": "6h", "availability": "auto", "source": MARS})
    migrated = migrate(old)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_recipe(migrated)
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations, [str(w.message) for w in deprecations]


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            obj()
            print(f"{name}: ok")
