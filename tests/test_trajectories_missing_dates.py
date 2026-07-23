# (C) Copyright 2025-2026 Anemoi contributors.
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


# ---------------------------------------------------------------------------
# Build side: TrajectoryGroups with missing
# ---------------------------------------------------------------------------


class TestTrajectoryGroupsMissing:

    def _make(self, missing=None, group_by=None, **base_dates_kwargs):
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
            group_by=group_by,
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

    def test_group_by_counts_base_dates(self):
        """``group_by`` counts base dates (whole trajectories), not pairs."""
        groups = self._make(group_by=1)
        all_groups = list(groups)
        assert len(all_groups) == 4  # one group per base date
        assert len(groups) == 4
        for group in all_groups:
            # Each group carries every step of a single trajectory.
            assert len(group.items) == 2
            assert len({bt for _, bt in group.items}) == 1

    def test_group_by_with_missing_skips_empty_groups(self):
        groups = self._make(missing=["2024-01-02"], group_by=1)
        all_groups = list(groups)
        assert len(all_groups) == 3
        assert len(groups) == 3
        for group in all_groups:
            for _, basetime in group.items:
                assert basetime != datetime.datetime(2024, 1, 2)


# ---------------------------------------------------------------------------
# Read side: TrajectoriesZarrWithMissingDates
# ---------------------------------------------------------------------------


class TestTrajectoriesZarrWithMissingDates:

    def setup_method(self):
        from anemoi.datasets.usage.trajectories.store import TrajectoriesZarrWithMissingDates

        self.group = make_trajectories_zarr(n_dates=6, n_steps=3, n_vars=2, n_cells=5, frequency_h=6)
        self.bd = np.array(self.group["base_dates"])
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
