# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for missing-date handling on the trajectories layout.

Build side:
- ``weekday`` / ``date`` recipe aliases on ``base_dates``.
- Explicit ``missing:`` list filtering through ``TrajectoryGroups``.
- ``TrajectoryFilter`` unit behaviour.

Read side:
- ``TrajectoriesZarrWithMissingDates`` raises on missing indices.
- ``open_dataset`` ``set_missing_dates`` / ``skip_missing_dates`` /
  ``fill_missing_dates`` work end-to-end on a synthetic 5-D zarr.
"""

import datetime

import numpy as np
import pytest

from test_trajectories import make_trajectories_zarr
from test_trajectories import open_trajectories_zarr


# ---------------------------------------------------------------------------
# Build side: TrajectoryFilter
# ---------------------------------------------------------------------------


class TestTrajectoryFilter:

    def test_filters_pairs_by_basetime(self):
        from anemoi.datasets.dates.groups import TrajectoryFilter

        bt1 = datetime.datetime(2024, 1, 1)
        bt2 = datetime.datetime(2024, 1, 2)
        st = datetime.timedelta(hours=6)

        f = TrajectoryFilter([bt2])
        kept = f([(bt1, st), (bt2, st), (bt1, st * 2)])
        assert kept == [(bt1, st), (bt1, st * 2)]

    def test_empty_missing_keeps_all(self):
        from anemoi.datasets.dates.groups import TrajectoryFilter

        bt = datetime.datetime(2024, 1, 1)
        st = datetime.timedelta(hours=6)

        f = TrajectoryFilter([])
        pairs = [(bt, st)]
        assert f(pairs) == pairs


# ---------------------------------------------------------------------------
# Build side: recipe aliases
# ---------------------------------------------------------------------------


class TestDatesAliases:

    def test_weekday_alias(self):
        from anemoi.datasets.dates import DatesProvider

        p = DatesProvider.from_config(
            start="2024-01-01",
            end="2024-01-31",
            frequency="1d",
            weekday=["tuesday", "friday"],
        )
        weekdays = {d.weekday() for d in p.values}
        # Tuesday is 1, Friday is 4
        assert weekdays == {1, 4}

    def test_weekday_alias_collision(self):
        from anemoi.datasets.dates import DatesProvider

        with pytest.raises(ValueError, match="weekday.*day_of_week"):
            DatesProvider.from_config(
                start="2024-01-01",
                end="2024-01-10",
                frequency="1d",
                weekday=["tuesday"],
                day_of_week=["friday"],
            )

    def test_weekday_unknown(self):
        from anemoi.datasets.dates import DatesProvider

        with pytest.raises(ValueError, match="Unknown weekday"):
            DatesProvider.from_config(
                start="2024-01-01",
                end="2024-01-10",
                frequency="1d",
                weekday=["tuesdai"],
            )

    def test_date_wildcard_first_of_month(self):
        from anemoi.datasets.dates import DatesProvider

        p = DatesProvider.from_config(
            start="2024-01-01",
            end="2024-12-31",
            frequency="1d",
            date="????-??-01",
        )
        days = {d.day for d in p.values}
        assert days == {1}
        assert len(p.values) == 12

    def test_date_wildcard_list_of_days(self):
        from anemoi.datasets.dates import DatesProvider

        p = DatesProvider.from_config(
            start="2024-01-01",
            end="2024-12-31",
            frequency="1d",
            date=["????-??-01", "????-??-15"],
        )
        days = {d.day for d in p.values}
        assert days == {1, 15}

    def test_date_wildcard_month(self):
        from anemoi.datasets.dates import DatesProvider

        p = DatesProvider.from_config(
            start="2024-01-01",
            end="2024-12-31",
            frequency="1d",
            date="????-06-??",
        )
        months = {d.month for d in p.values}
        assert months == {6}
        assert len(p.values) == 30  # June 2024

    def test_date_wildcard_rejects_fixed_year(self):
        from anemoi.datasets.dates import DatesProvider

        with pytest.raises(ValueError, match=r"\?\?\?\?"):
            DatesProvider.from_config(
                start="2024-01-01",
                end="2024-12-31",
                frequency="1d",
                date="2024-??-01",
            )

    def test_month_alias(self):
        from anemoi.datasets.dates import DatesProvider

        p = DatesProvider.from_config(
            start="2024-01-01",
            end="2024-12-31",
            frequency="1d",
            month=[6, 7],
        )
        months = {d.month for d in p.values}
        assert months == {6, 7}


# ---------------------------------------------------------------------------
# Build side: TrajectoryGroups with missing
# ---------------------------------------------------------------------------


class TestTrajectoryGroupsMissing:

    def _make(self, missing=None, **base_dates_kwargs):
        from anemoi.datasets.dates.groups import TrajectoryGroups

        base_dates = {
            "start": "2024-01-01",
            "end": "2024-01-04",
            "frequency": "1d",
        }
        base_dates.update(base_dates_kwargs)
        if missing is not None:
            base_dates["missing"] = missing

        return TrajectoryGroups(
            steps={"start": "6h", "end": "12h", "frequency": "6h"},
            group_by=None,
            base_dates=base_dates,
        )

    def test_no_missing_yields_all_pairs(self):
        groups = self._make()
        # 4 base dates × 2 steps = 8 pairs
        all_pairs = list(groups)
        assert len(all_pairs) == 1  # one group (group_by=None)
        assert len(all_pairs[0].items) == 8

    def test_missing_filters_basetime_pairs(self):
        groups = self._make(missing=["2024-01-02"])
        all_pairs = list(groups)
        # 3 base dates × 2 steps = 6 pairs
        assert len(all_pairs[0].items) == 6
        # 2024-01-02 should not appear as a basetime in any kept pair
        for valid_time, basetime in all_pairs[0].items:
            assert basetime != datetime.datetime(2024, 1, 2)

    def test_provider_missing_is_basetime_list(self):
        groups = self._make(missing=["2024-01-02"])
        assert datetime.datetime(2024, 1, 2) in groups.provider.missing

    def test_factorise_keeps_slot_for_missing(self):
        """The on-disk slot for a missing base date stays in the array."""
        groups = self._make(missing=["2024-01-02"])
        basetimes, _ = groups.provider.factorise()
        assert datetime.datetime(2024, 1, 2) in basetimes
        assert len(basetimes) == 4


# ---------------------------------------------------------------------------
# Read side: TrajectoriesZarrWithMissingDates
# ---------------------------------------------------------------------------


class TestTrajectoriesZarrWithMissingDates:

    def setup_method(self):
        from anemoi.datasets.usage.trajectories.store import TrajectoriesZarrWithMissingDates

        self.group = make_trajectories_zarr(n_dates=6, n_steps=3, n_vars=2, n_cells=5, frequency_h=6)
        self.bd = np.array(self.group.base_dates)
        # Mark indices 1 and 4 as missing in the store
        self.group.attrs["missing_dates"] = [str(self.bd[1]), str(self.bd[4])]

        self.ds = open_trajectories_zarr(self.group).mutate()
        assert isinstance(self.ds, TrajectoriesZarrWithMissingDates)

    def test_missing_indices(self):
        assert self.ds.missing == {1, 4}

    def test_int_index_on_present_date(self):
        arr = self.ds[0]
        assert arr.shape == (2, 1, 3, 5)

    def test_int_index_on_missing_raises(self):
        from anemoi.datasets import MissingDateError

        with pytest.raises(MissingDateError):
            self.ds[1]

    def test_slice_overlapping_missing_raises(self):
        from anemoi.datasets import MissingDateError

        with pytest.raises(MissingDateError):
            self.ds[0:3]

    def test_slice_skipping_missing_ok(self):
        # indices 2,3 are present
        arr = self.ds[2:4]
        assert arr.shape == (2, 2, 1, 3, 5)

    def test_tuple_index_with_int_first(self):
        from anemoi.datasets import MissingDateError

        with pytest.raises(MissingDateError):
            self.ds[1, :]

        arr = self.ds[0, :]
        assert arr.shape == (2, 1, 3, 5)

    def test_mutate_idempotent(self):
        assert self.ds.mutate() is self.ds


# ---------------------------------------------------------------------------
# Read side: set_missing_dates / skip_missing_dates / fill_missing_dates
# ---------------------------------------------------------------------------


class TestSubsetMissingKwargs:

    def setup_method(self):
        self.group = make_trajectories_zarr(n_dates=6, n_steps=3, n_vars=2, n_cells=5, frequency_h=6)
        self.ds_raw = open_trajectories_zarr(self.group)

    def test_set_missing_dates(self):
        from anemoi.datasets import MissingDateError

        bd = np.array(self.group.base_dates)
        ds = self.ds_raw._subset(set_missing_dates=[str(bd[2])])
        assert ds.missing == {2}

        with pytest.raises(MissingDateError):
            ds[2]

    def test_skip_missing_dates(self):
        bd = np.array(self.group.base_dates)
        # Tag index 2 as missing, then iterate with expected_access=2
        ds = self.ds_raw._subset(set_missing_dates=[str(bd[2])])
        ds = ds._subset(skip_missing_dates=True, expected_access=2)

        # Iterate — must not raise
        seen = 0
        for sample in ds:
            assert isinstance(sample, tuple) and len(sample) == 2
            seen += 1
        assert seen == len(ds)

    def test_fill_missing_dates_closest(self):
        # Mark index 2 missing in the store, then open with fill
        bd = np.array(self.group.base_dates)
        self.group.attrs["missing_dates"] = [str(bd[2])]

        ds_full = open_trajectories_zarr(self.group)
        ds = ds_full.mutate()._subset(fill_missing_dates="closest")

        # closest defaults to "up" → index 3
        np.testing.assert_array_equal(ds[2], ds_full[3])

    def test_fill_missing_dates_interpolate(self):
        bd = np.array(self.group.base_dates)
        self.group.attrs["missing_dates"] = [str(bd[2])]

        ds_full = open_trajectories_zarr(self.group)
        ds = ds_full.mutate()._subset(fill_missing_dates="interpolate")

        expected = 0.5 * ds_full[1] + 0.5 * ds_full[3]
        np.testing.assert_allclose(ds[2], expected)
