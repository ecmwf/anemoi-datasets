# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

from anemoi.datasets.create.config import Config, LoadersConfig, set_to_test_mode
from anemoi.datasets.dates.groups import Groups

HERE = os.path.dirname(__file__)


def _load_config(name):
    path = os.path.join(HERE, name)
    return Config(path)


def test_set_to_test_mode_limits_dates_with_real_recipe():
    cfg = _load_config("concat.yaml")

    # Original config: 2020-12-30 00:00 to 2021-01-03 12:00 at 12h = 10 dates
    original_groups = Groups(**LoadersConfig(cfg).dates)
    original_dates = original_groups.provider.values
    assert len(original_dates) == 10

    set_to_test_mode(cfg)

    # After test mode, should produce exactly 4 dates
    test_groups = Groups(**cfg["dates"])
    test_dates = test_groups.provider.values
    assert len(test_dates) == 4
    assert cfg["dates"]["group_by"] == 4


def test_set_to_test_mode_reduces_grid_and_ensemble():
    cfg = Config(
        {
            "dates": {"start": "2020-12-30 00:00:00", "end": "2021-01-03 12:00:00", "frequency": "12h"},
            "input": {
                "mars": {
                    "grid": "0.25/0.25",
                    "number": [0, 1, 2, 3, 4, 5],
                    "param": ["2t"],
                }
            },
        }
    )

    set_to_test_mode(cfg)

    assert cfg["input"]["mars"]["grid"] == "20./20."
    assert cfg["input"]["mars"]["number"] == [0, 1, 2]

