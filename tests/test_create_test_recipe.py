# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from anemoi.utils.cli import cli_main

from anemoi.datasets import __version__
from anemoi.datasets.commands import COMMANDS
from anemoi.datasets.create.recipe import Recipe


HERE = Path(__file__).parent / "create"


def _as_timestamp(value) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


@pytest.fixture
def recipe_path() -> Path:
    return HERE / "concat.yaml"


def test_create_test_recipe_includes_last_dates(recipe_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "concat-test.yaml"

    cli_main(
        __version__,
        "anemoi-datasets",
        COMMANDS,
        [
            "create-test-recipe",
            str(recipe_path),
            str(output_path),
            "--n-dates",
            "2",
            "--n-levels",
            "2",
        ],
    )

    data = yaml.safe_load(output_path.read_text())

    assert data["description"].startswith(
        "This is a test version of the following recipe, created using "
        "anemoi-datasets create-test-recipe "
        "--dates --last-dates --grid --level --n-dates 2 --n-levels 2:"
    )

    assert _as_timestamp(data["dates"]["start"]) == "2020-12-30 00:00:00"
    assert _as_timestamp(data["dates"]["end"]) == "2021-01-03 12:00:00"
    assert data["dates"]["frequency"] == "12h"
    assert data["dates"]["missing"] == [
        {
            "start": "2020-12-31 00:00:00",
            "end": "2021-01-02 12:00:00",
        }
    ]
    assert _as_timestamp(data["input"]["concat"][0]["dates"]["start"]) == "2020-12-30 00:00:00"
    assert _as_timestamp(data["input"]["concat"][0]["dates"]["end"]) == "2021-01-01 12:00:00"
    assert data["input"]["concat"][0]["dates"]["frequency"] == "12h"
    assert data["input"]["concat"][0]["dates"]["missing"] == [
        {
            "start": "2020-12-31 00:00:00",
            "end": "2020-12-31 12:00:00",
        }
    ]


def test_recipe_supports_missing_ranges() -> None:
    recipe = Recipe(
        dates={
            "start": "2020-01-01 00:00:00",
            "end": "2020-01-01 18:00:00",
            "frequency": "6h",
            "missing": [
                {
                    "start": "2020-01-01 06:00:00",
                    "end": "2020-01-01 12:00:00",
                }
            ],
        },
        input={"join": []},
    )

    assert [d.strftime("%Y-%m-%d %H:%M:%S") for d in recipe.dates.values] == [
        "2020-01-01 00:00:00",
        "2020-01-01 18:00:00",
    ]


def test_create_test_recipe_rejects_last_dates_with_no_dates(recipe_path: Path, tmp_path: Path) -> None:
    parser = ArgumentParser(prog="anemoi-datasets create-test-recipe")
    command = COMMANDS["create-test-recipe"]
    command.add_arguments(parser)

    args = parser.parse_args([str(recipe_path), str(tmp_path / "out.yaml"), "--no-dates", "--last-dates"])

    with pytest.raises(SystemExit):
        command.check(parser, args)


def test_create_test_recipe_cli_rejects_last_dates_with_no_dates(recipe_path: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "out.yaml"

    with pytest.raises(SystemExit):
        cli_main(
            __version__,
            "anemoi-datasets",
            COMMANDS,
            [
                "create-test-recipe",
                str(recipe_path),
                str(output_path),
                "--no-dates",
                "--last-dates",
            ],
        )

    assert not output_path.exists()