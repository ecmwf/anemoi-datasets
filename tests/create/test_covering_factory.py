# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the recipe-side covering_factory dispatch and back-compat."""

import datetime

import pytest

from anemoi.datasets.create.sources.windowed.covering import AutoCovering
from anemoi.datasets.create.sources.windowed.covering import covering_factory
from anemoi.datasets.create.sources.windowed.subwindows import contributions_of


def _intervals(covering, *args, **kwargs):
    """The archived intervals a covering asks for, flattened out of its subwindows.

    Most of these tests are about *which fields are retrieved*, which is what the
    contributions are; the subwindow structure has its own tests.
    """
    return contributions_of(covering.partition(*args, **kwargs))


def test_discriminator_auto():
    """covering: { auto: <X> } returns an AutoCovering over X."""
    sel = covering_factory({"auto": [(0, "0-6/0-12"), (12, "0-6/0-12")]})
    assert isinstance(sel, AutoCovering)


def test_legacy_list_treated_as_auto():
    """A bare list (legacy availability:) is treated as the auto value."""
    sel = covering_factory([(0, "0-6/0-12"), (12, "0-6/0-12")])
    assert isinstance(sel, AutoCovering)


def test_legacy_mars_dict_treated_as_auto():
    """A bare mars dict (legacy availability:) is treated as the auto value."""
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
    """Recipe migrator rewrites accumulate.availability to a description key.

    (Full migration coverage lives in test_recipe_migrate.py.)
    """
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
    assert "covering" not in block
    assert "type" not in block["from"]
    assert set(block["from"]) == {"base_dates", "steps", "accumulation"}
    assert block["from"]["accumulation"] == "from-zero"


# ---------------------------------------------------------------------------
# lookup-table: the entry pins WHICH intervals may be used, the signed search
# decides HOW they combine — so a covering that does not add up is impossible.
# ---------------------------------------------------------------------------


def _lookup(**table):
    from anemoi.datasets.create.sources.windowed.covering import AutoCovering
    from anemoi.datasets.create.sources.windowed.interval_generators import LookupTableIntervalGenerator

    return AutoCovering(LookupTableIntervalGenerator(start="1970-01-01", **table))


def _window(h0, h1):
    return datetime.datetime(2024, 1, 1, h0), datetime.datetime(2024, 1, 1, h1)


@pytest.mark.parametrize("steps", ["0-12/0-6", "0-6/0-12"])
def test_lookup_table_expresses_a_from_zero_difference(steps):
    """An entry naming two from-zero fields is combined as a(0,12) - a(0,6), not summed.

    Regression: these were previously returned as two *positive* intervals and
    silently added, giving 18h of accumulation labelled as a 6h window.
    """
    start, end = _window(6, 12)
    cover = _intervals(_lookup(**{"6-12": [0, steps]}), start, end)

    assert sum(i.length for i in cover) == (end - start).total_seconds()
    signed = {
        (int((i.min - i.base).total_seconds() // 3600), int((i.max - i.base).total_seconds() // 3600)): i.sign
        for i in cover
    }
    assert signed == {(0, 12): 1, (0, 6): -1}


def test_lookup_table_rejects_entries_that_cannot_cover_the_window():
    start, end = _window(6, 12)
    with pytest.raises(ValueError, match="Cannot find coverage"):
        _intervals(_lookup(**{"6-12": [0, "0-12/0-3"]}), start, end)


def test_lookup_table_single_archived_window_still_works():
    """The documented form: the archive natively stores the requested windows."""
    start, end = _window(6, 12)
    cover = _intervals(
        _lookup(**{"0-6": [18, "6-12"], "6-12": [18, "12-18"], "12-18": [18, "18-24"], "18-24": [18, "0-6"]}),
        start,
        end,
    )
    assert len(cover) == 1
    assert cover[0].sign == 1
    assert sum(i.length for i in cover) == (end - start).total_seconds()


def test_a_subwindow_whose_contributions_do_not_rebuild_it_is_rejected():
    """The check that used to be check_covering, now per subwindow rather than per window.

    Two positive from-zero fields summed give 18h of accumulation labelled as a 6h
    window; one of them has to be subtracted.
    """
    from anemoi.datasets.create.intervals import SignedInterval
    from anemoi.datasets.create.sources.windowed.subwindows import Subwindow
    from anemoi.datasets.create.sources.windowed.subwindows import validate_contributions

    bt = datetime.datetime(2024, 1, 1, 0)
    start, end = _window(6, 12)
    both_positive = Subwindow(
        interval=SignedInterval(start, end),
        contributions=(
            SignedInterval(bt, bt + datetime.timedelta(hours=12), base=bt),
            SignedInterval(bt, bt + datetime.timedelta(hours=6), base=bt),
        ),
    )
    with pytest.raises(ValueError, match="do not reconstruct it"):
        validate_contributions(both_positive)

    # the same two fields, one of them subtracted, do rebuild it
    validate_contributions(
        Subwindow(
            interval=SignedInterval(start, end),
            contributions=(
                SignedInterval(bt, bt + datetime.timedelta(hours=12), base=bt),
                -SignedInterval(bt, bt + datetime.timedelta(hours=6), base=bt),
            ),
        )
    )
