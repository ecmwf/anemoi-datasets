# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime

import pytest

from anemoi.datasets.commands.recipe.migrate import (
    fix_datetimes,
    migrate,
    migrate_accumulations,
    remove_useless_common_block,
)


def test_migrate_accumulations_with_int_period():
    """Test migration of accumulations source with integer accumulation_period."""
    config = {
        "accumulations": {
            "accumulation_period": 6,
            "param": "tp",
            "class": "od",
            "stream": "oper",
        }
    }
    result = migrate_accumulations(config)
    assert "accumulate" in result
    assert "accumulations" not in result
    assert result["accumulate"]["period"] == 6
    assert result["accumulate"]["availability"] == "auto"
    assert result["accumulate"]["source"]["mars"]["param"] == "tp"


def test_migrate_accumulations_with_default_period(caplog):
    """Test migration with default accumulation_period and warning."""
    config = {
        "accumulations": {
            "param": "tp",
            "class": "od",
            "stream": "oper",
        }
    }
    result = migrate_accumulations(config)
    assert "accumulate" in result
    assert result["accumulate"]["period"] == 6
    # Check that warning was logged
    assert "using default value of 6 hours" in caplog.text


def test_migrate_accumulations_with_tuple_period():
    """Test migration of accumulations source with tuple accumulation_period."""
    config = {
        "accumulations": {
            "accumulation_period": [0, 12],
            "param": "tp",
            "class": "od",
            "stream": "oper",
        }
    }
    result = migrate_accumulations(config)
    assert "accumulate" in result
    assert result["accumulate"]["period"] == 12
    assert result["accumulate"]["availability"] == [[0, ["0-12"]], [6, ["0-12"]], [12, ["0-12"]], [18, ["0-12"]]]


def test_migrate_accumulations_with_tuple_period_non_zero():
    """Test migration with non-zero start in accumulation_period tuple."""
    config = {
        "accumulations": {
            "accumulation_period": [6, 12],
            "param": "tp",
            "class": "od",
            "stream": "oper",
        }
    }
    result = migrate_accumulations(config)
    assert result["accumulate"]["period"] == 6
    assert result["accumulate"]["availability"] == [[0, ["0-6", "0-12"]], [6, ["0-6", "0-12"]], [12, ["0-6", "0-12"]], [18, ["0-6", "0-12"]]]


def test_migrate_accumulations_with_time_field(caplog):
    """Test migration with time field that gets converted to base times."""
    config = {
        "accumulations": {
            "accumulation_period": [0, 12],
            "param": "tp",
            "class": "od",
            "stream": "oper",
            "time": ["00:00", "12:00"],
        }
    }
    result = migrate_accumulations(config)
    assert result["accumulate"]["availability"] == [[0, ["0-12"]], [12, ["0-12"]]]
    assert "time" not in result["accumulate"]["source"]["mars"]
    assert "using time values [0, 12]" in caplog.text


def test_migrate_accumulations_with_od_enfo(caplog):
    """Test migration with od/enfo class/stream combination."""
    config = {
        "accumulations": {
            "accumulation_period": 6,
            "param": "tp",
            "class": "od",
            "stream": "enfo",
        }
    }
    result = migrate_accumulations(config)
    # Should use explicit availability instead of 'auto'
    assert result["accumulate"]["availability"] == [[0, ["0-6"]], [6, ["0-6"]], [12, ["0-6"]], [18, ["0-6"]]]
    assert "not yet supported for class=od stream=enfo" in caplog.text


def test_migrate_accumulations_strips_step(caplog):
    """Test that 'step' field is removed with a warning."""
    config = {
        "accumulations": {
            "accumulation_period": 6,
            "param": "tp",
            "step": "6/12/18",
        }
    }
    result = migrate_accumulations(config)
    assert "step" not in result["accumulate"]["source"]["mars"]
    assert "Stripping 'step:" in caplog.text


def test_migrate_accumulations_strips_reset_frequency(caplog):
    """Test that 'accumulations_reset_frequency' field is removed with a warning."""
    config = {
        "accumulations": {
            "accumulation_period": 6,
            "param": "tp",
            "accumulations_reset_frequency": 24,
        }
    }
    result = migrate_accumulations(config)
    assert "accumulations_reset_frequency" not in result["accumulate"]["source"]["mars"]
    assert "Stripping 'accumulations_reset_frequency:" in caplog.text


def test_migrate_accumulations_changes_type_an_to_fc(caplog):
    """Test that type 'an' is changed to 'fc' with a warning."""
    config = {
        "accumulations": {
            "accumulation_period": 6,
            "param": "tp",
            "type": "an",
        }
    }
    result = migrate_accumulations(config)
    assert result["accumulate"]["source"]["mars"]["type"] == "fc"
    assert "Changing 'type: an' to 'type: fc'" in caplog.text


def test_migrate_accumulations_no_accumulations():
    """Test that configs without 'accumulations' are returned unchanged."""
    config = {
        "mars": {
            "param": "2t",
            "class": "od",
            "stream": "oper",
        }
    }
    result = migrate_accumulations(config)
    assert result == config


def test_migrate_accumulations_nested():
    """Test migration with nested accumulations in a list of sources."""
    config = {
        "input": [
            {
                "mars": {
                    "param": "2t",
                }
            },
            {
                "accumulations": {
                    "accumulation_period": 6,
                    "param": "tp",
                }
            },
        ]
    }
    result = migrate_accumulations(config)
    assert "mars" in result["input"][0]
    assert "accumulate" in result["input"][1]
    assert "accumulations" not in result["input"][1]


def test_migrate_accumulations_invalid_period():
    """Test that invalid accumulation_period raises ValueError."""
    config = {
        "accumulations": {
            "accumulation_period": "invalid",
            "param": "tp",
        }
    }
    with pytest.raises(ValueError, match="Invalid accumulation_period"):
        migrate_accumulations(config)


def test_fix_datetimes_datetime_objects():
    """Test conversion of datetime objects to plain strings."""
    config = {
        "start": datetime.datetime(2024, 1, 1, 0, 0, 0),
        "end": datetime.datetime(2024, 12, 31, 18, 0, 0),
    }
    result = fix_datetimes(config)
    assert result["start"] == "2024-01-01"
    assert result["end"] == "2024-12-31 18:00:00"


def test_fix_datetimes_date_objects():
    """Test conversion of date objects to plain strings."""
    config = {
        "start": datetime.date(2024, 1, 1),
    }
    result = fix_datetimes(config)
    assert result["start"] == "2024-01-01"


def test_fix_datetimes_nested():
    """Test conversion of nested datetime objects."""
    config = {
        "dates": {
            "start": datetime.datetime(2024, 1, 1, 12, 0, 0),
            "end": datetime.datetime(2024, 12, 31, 0, 0, 0),
        }
    }
    result = fix_datetimes(config)
    assert result["dates"]["start"] == "2024-01-01 12:00:00"
    assert result["dates"]["end"] == "2024-12-31"


def test_fix_datetimes_no_datetimes():
    """Test that configs without datetimes are returned unchanged."""
    config = {
        "param": "2t",
        "number": 42,
    }
    result = fix_datetimes(config)
    assert result == config


def test_remove_useless_common_block():
    """Test removal of 'common' key from config."""
    config = {
        "common": {"key": "value"},
        "mars": {"param": "2t"},
    }
    result = remove_useless_common_block(config)
    assert "common" not in result
    assert "mars" in result


def test_migrate_full_pipeline():
    """Test the full migrate pipeline with all transformations."""
    config = {
        "common": {"key": "value"},
        "start": datetime.datetime(2024, 1, 1, 0, 0, 0),
        "end": datetime.datetime(2024, 12, 31, 0, 0, 0),
        "accumulations": {
            "accumulation_period": 6,
            "param": "tp",
            "class": "od",
            "stream": "oper",
        },
    }
    result = migrate(config)
    # Check that all transformations were applied
    assert "common" not in result
    assert result["start"] == "2024-01-01"
    assert result["end"] == "2024-12-31"
    assert "accumulate" in result
    assert "accumulations" not in result


def test_migrate_no_changes():
    """Test that configs with no changes needed are returned unchanged."""
    config = {
        "mars": {
            "param": "2t",
            "class": "od",
            "stream": "oper",
        }
    }
    result = migrate(config)
    assert result == config


def test_migrate_accumulations_primitive_values():
    """Test that primitive values are returned unchanged (base case)."""
    assert migrate_accumulations("string") == "string"
    assert migrate_accumulations(42) == 42
    assert migrate_accumulations(3.14) == 3.14
    assert migrate_accumulations(True) is True
    assert migrate_accumulations(None) is None


def test_migrate_accumulations_empty_dict():
    """Test that empty dicts are handled correctly."""
    config = {}
    result = migrate_accumulations(config)
    assert result == {}


def test_migrate_accumulations_empty_list():
    """Test that empty lists are handled correctly."""
    config = []
    result = migrate_accumulations(config)
    assert result == []


def test_migrate_accumulations_list_of_primitives():
    """Test that lists of primitives are returned unchanged."""
    config = [1, 2, 3, "a", "b", "c"]
    result = migrate_accumulations(config)
    assert result == config
