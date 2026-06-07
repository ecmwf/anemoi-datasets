# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the recipe-side covering_factory dispatch and back-compat."""

import pytest

from anemoi.datasets.create.sources.accumulate.covering import AutoCovering
from anemoi.datasets.create.sources.accumulate.covering import covering_factory


def test_discriminator_auto():
    """covering: { auto: <X> } returns an AutoCovering over X."""
    sel = covering_factory({"auto": [(0, "0-6/0-12"), (12, "0-6/0-12")]})
    assert isinstance(sel, AutoCovering)


def test_eager_list_treated_as_auto():
    """A bare list (eager availability:) is treated as the auto value."""
    sel = covering_factory([(0, "0-6/0-12"), (12, "0-6/0-12")])
    assert isinstance(sel, AutoCovering)


def test_eager_mars_dict_treated_as_auto():
    """A bare mars dict (eager availability:) is treated as the auto value."""
    sel = covering_factory({"mars": {"class": "ea", "stream": "oper"}})
    assert isinstance(sel, AutoCovering)


def test_discriminator_forecast_rejected():
    """The forecast branch is implicit; explicit declaration is rejected with a clear error."""
    with pytest.raises(ValueError, match="trajectory branch is selected implicitly"):
        covering_factory({"forecast": {}})


def test_discriminator_cycle_not_implemented():
    with pytest.raises(NotImplementedError, match="cycle"):
        covering_factory({"cycle": {}})


def test_migrate_rewrites_availability():
    """Recipe migrator rewrites accumulate.availability to covering.auto."""
    from anemoi.datasets.commands.recipe.migrate import migrate

    old = {
        "input": {
            "join": [
                {
                    "accumulate": {
                        "period": "6h",
                        "availability": [(0, "0-6/0-12"), (12, "0-6/0-12")],
                        "source": {"mars": {"class": "od"}},
                    }
                }
            ]
        }
    }
    new = migrate(old)
    block = new["input"]["join"][0]["accumulate"]
    assert "availability" not in block
    assert block["covering"] == {"auto": [(0, "0-6/0-12"), (12, "0-6/0-12")]}


def test_migrate_leaves_existing_covering_untouched():
    from anemoi.datasets.commands.recipe.migrate import migrate

    old = {
        "input": {
            "join": [
                {
                    "accumulate": {
                        "period": "6h",
                        "covering": {"auto": "auto"},
                        "source": {"mars": {"class": "od"}},
                    }
                }
            ]
        }
    }
    new = migrate(old)
    block = new["input"]["join"][0]["accumulate"]
    assert block["covering"] == {"auto": "auto"}
    assert "availability" not in block
