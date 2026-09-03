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
from types import SimpleNamespace

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

    def test_group_entirely_missing_is_dropped(self):
        """A fixed-size batch whose every base date is missing yields no group.

        With ``group_by=2`` the raw batches are [01,02], [03,04], [05,06]; the
        middle batch is entirely missing, so it is dropped rather than becoming
        an all-missing group. This is why the statistics never see a group with
        no real rows.
        """
        groups = self._make(start="2024-01-01", end="2024-01-06", group_by=2, missing=["2024-01-03", "2024-01-04"])
        all_groups = list(groups)
        assert len(groups) == 2
        assert len(all_groups) == 2
        seen = {bt for group in all_groups for _, bt in group.items}
        assert datetime.datetime(2024, 1, 3) not in seen
        assert datetime.datetime(2024, 1, 4) not in seen


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


# ---------------------------------------------------------------------------
# Build side: per-group precomputed statistics with scattered / missing rows
# ---------------------------------------------------------------------------


class _StubDataset:
    """Minimal stand-in exposing the attributes ``load_precomputed`` reads."""

    def __init__(self, base_dates, missing_rows):
        self.base_dates = np.asarray(base_dates).astype("datetime64[s]")
        # ``missing_dates`` metadata is a list of ISO date strings, one per
        # missing base-date slot -- mirror that here from the missing row indices.
        self._missing_dates = [str(self.base_dates[i]) for i in missing_rows]

    @property
    def data(self):
        return SimpleNamespace(shape=(len(self.base_dates), 2, 1, 4, 3))

    def get_metadata(self, key, default=None):
        return self._missing_dates if key == "missing_dates" else default


class TestTrajectoryPrecomputedStatistics:
    """The per-group precomputed statistics must be correct even when a group's
    rows are scattered across axis 0 (non-contiguous ``group_by`` and/or missing
    base-date slots), and finalisation must be a plain merge over recorded
    indices.
    """

    N_ROWS = 12
    MISSING_ROWS = (2, 5, 9)  # NaN slots kept in the array
    NAMES = ["a", "b"]
    TENDENCIES = {"1step": 1}

    def _data(self):
        rng = np.arange(self.N_ROWS * 2 * 1 * 4 * 3, dtype=np.float64)
        data = rng.reshape(self.N_ROWS, 2, 1, 4, 3)
        data[list(self.MISSING_ROWS)] = np.nan
        return data

    def _collector(self, filter=None):
        from anemoi.datasets.create.statistics import TrajectoryStatisticsCollector

        return TrajectoryStatisticsCollector(
            variables_names=self.NAMES, filter=filter, tendencies=dict(self.TENDENCIES)
        )

    def _base_dates(self):
        return (np.datetime64("2003-01-01", "s") + np.arange(self.N_ROWS) * np.timedelta64(1, "D")).astype(
            "datetime64[s]"
        )

    def _collect_rows(self, data, indices, filter=None):
        collector = self._collector(filter=filter)
        base_dates = self._base_dates()
        for i in indices:
            collector.collect(data[i : i + 1], base_dates[i : i + 1])
        return collector

    def _groups(self):
        # A partition of the non-missing rows into scattered groups, mimicking
        # an ``MMDD`` grouping where each group picks rows spread across the axis.
        return [[0, 6, 11], [1, 7, 10], [3, 4, 8]]

    def _serialise_groups(self, tmp_path, data, groups, filter=None):
        paths = []
        for g, indices in enumerate(groups):
            collector = self._collect_rows(data, indices, filter=filter)
            path = str(tmp_path / f"statistics_{g:06d}.pkl")
            collector.serialise(
                path, group=g, start=min(indices), end=max(indices) + 1, indices=np.array(indices, dtype=np.int64)
            )
            paths.append(path)
        return paths

    def test_merge_matches_full_reference(self, tmp_path):
        from anemoi.datasets.create.statistics import TrajectoryStatisticsCollector

        data = self._data()
        groups = self._groups()
        paths = self._serialise_groups(tmp_path, data, groups)

        ds = _StubDataset(self._base_dates(), self.MISSING_ROWS)
        merged = TrajectoryStatisticsCollector.load_precomputed(ds, paths)

        # Reference: collect over every non-missing row in one pass.
        non_missing = [i for i in range(self.N_ROWS) if i not in self.MISSING_ROWS]
        reference = self._collect_rows(data, non_missing)

        got = merged.statistics()
        expected = reference.statistics()
        assert set(got) == set(expected)
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-12, atol=1e-12, err_msg=key)
        # Tendency statistics must be present and finite.
        assert any(k.startswith("statistics_tendencies_1step_") for k in got)

    def test_merge_matches_full_reference_with_statistics_envelope(self, tmp_path):
        """Same, but with a statistics-date envelope that excludes some rows.

        The envelope drops out-of-window trajectories *inside* ``collect``; the
        coverage check still requires every non-missing row to be recorded
        (``load_done`` gathers a group's rows regardless of the envelope), and
        the merged statistics must match a full-array reference under the same
        filter -- proving per-row/scattered evaluation reproduces the batch
        result even when only a subset of rows contributes.
        """
        from anemoi.datasets.create.recipe.statistics import TrajectoryStatisticsFilter
        from anemoi.datasets.create.statistics import TrajectoryStatisticsCollector

        base_dates = self._base_dates()
        zero = np.timedelta64(0, "s")
        # Envelope [rows 3 .. 8]; step_start=step_end=0 so the base date alone
        # decides. In-window non-missing rows are {3, 4, 6, 7, 8} (5 is missing).
        filt = TrajectoryStatisticsFilter(base_dates[3], base_dates[8], zero, zero)

        data = self._data()
        groups = self._groups()
        paths = self._serialise_groups(tmp_path, data, groups, filter=filt)

        ds = _StubDataset(base_dates, self.MISSING_ROWS)
        merged = TrajectoryStatisticsCollector.load_precomputed(ds, paths)

        non_missing = [i for i in range(self.N_ROWS) if i not in self.MISSING_ROWS]
        reference = self._collect_rows(data, non_missing, filter=filt)

        got = merged.statistics()
        expected = reference.statistics()
        assert set(got) == set(expected)
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-12, atol=1e-12, err_msg=key)

    def test_boundary_missing_and_envelope_edges(self, tmp_path):
        """First row missing, last row missing, and envelope edges on real rows.

        - row 0 (first) and row 11 (last) are missing -> must be absent from both
          the expected and the covered index sets, so coverage still matches;
        - the envelope keeps base dates in ``[d2 .. d9]`` (inclusive): rows 2 and
          9 sit exactly on the bounds, rows 1 and 10 are non-missing but just
          outside, so they are covered (gathered) yet contribute nothing.
        """
        from anemoi.datasets.create.recipe.statistics import TrajectoryStatisticsFilter
        from anemoi.datasets.create.statistics import TrajectoryStatisticsCollector

        base_dates = self._base_dates()
        missing_rows = (0, 6, 11)
        data = np.arange(self.N_ROWS * 2 * 1 * 4 * 3, dtype=np.float64).reshape(self.N_ROWS, 2, 1, 4, 3)
        data[list(missing_rows)] = np.nan

        zero = np.timedelta64(0, "s")
        filt = TrajectoryStatisticsFilter(base_dates[2], base_dates[9], zero, zero)

        non_missing = [i for i in range(self.N_ROWS) if i not in missing_rows]
        # Scatter the non-missing rows across three groups.
        groups = [non_missing[0::3], non_missing[1::3], non_missing[2::3]]
        paths = self._serialise_groups(tmp_path, data, groups, filter=filt)

        ds = _StubDataset(base_dates, missing_rows)
        merged = TrajectoryStatisticsCollector.load_precomputed(ds, paths)  # coverage must pass

        reference = self._collect_rows(data, non_missing, filter=filt)
        got = merged.statistics()
        expected = reference.statistics()
        assert set(got) == set(expected)
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-12, atol=1e-12, err_msg=key)

    def test_duplicate_rows_rejected(self, tmp_path):
        from anemoi.datasets.create.statistics import TrajectoryStatisticsCollector

        data = self._data()
        groups = [[0, 6, 11], [6, 7, 10], [3, 4, 8]]  # row 6 covered twice
        paths = self._serialise_groups(tmp_path, data, groups)
        ds = _StubDataset(self._base_dates(), self.MISSING_ROWS)
        with pytest.raises(ValueError, match="more than one group"):
            TrajectoryStatisticsCollector.load_precomputed(ds, paths)

    def test_incomplete_coverage_rejected(self, tmp_path):
        from anemoi.datasets.create.statistics import TrajectoryStatisticsCollector

        data = self._data()
        groups = [[0, 6, 11], [1, 7, 10], [3, 4]]  # row 8 never covered
        paths = self._serialise_groups(tmp_path, data, groups)
        ds = _StubDataset(self._base_dates(), self.MISSING_ROWS)
        with pytest.raises(ValueError, match="non-missing rows"):
            TrajectoryStatisticsCollector.load_precomputed(ds, paths)

    def test_missing_group_rejected(self, tmp_path):
        from anemoi.datasets.create.statistics import TrajectoryStatisticsCollector

        data = self._data()
        paths = self._serialise_groups(tmp_path, data, self._groups())
        del paths[1]  # drop group 1
        ds = _StubDataset(self._base_dates(), self.MISSING_ROWS)
        with pytest.raises(ValueError, match="Missing statistics for group 1"):
            TrajectoryStatisticsCollector.load_precomputed(ds, paths)


# ---------------------------------------------------------------------------
# End-to-end: real creator code (both statistics branches) on scattered MMDD
# groups with missing base-date slots. Only the data *loading* is bypassed --
# the zarr is pre-populated -- so load_done / _group_row_indices /
# _compute_partial_statistics{,_indices} / serialise / load_precomputed are all
# the production paths.
# ---------------------------------------------------------------------------


class TestTrajectoryStatisticsEndToEnd:

    @pytest.fixture(autouse=True)
    def _restore_seterr(self):
        # ``Creator.__init__`` flips numpy's global error mode to ``raise``;
        # restore it so it does not leak into other tests.
        old = np.geterr()
        yield
        np.seterr(**old)

    def _make_recipe(self, missing):
        from anemoi.datasets.create.recipe import Recipe

        return Recipe(
            base_dates={"start": "2020-01-01", "end": "2021-01-31", "frequency": "1d", "missing": missing},
            input={"mars": {"param": ["t"]}},
            output={"layout": "trajectories"},
            steps={"start": "6h", "end": "12h", "frequency": "6h"},
            statistics={"tendencies": True},
            build={"group_by": "MMDD"},
        )

    def _build(self, tmp_path):
        """Build a recipe + a matching pre-populated dataset and return a creator.

        Base dates are daily over two years; only 1--5 January of 2020 and 2021
        are kept, all other days are missing. With ``group_by: MMDD`` this yields
        five groups, each holding the same calendar day in both years -- rows
        ``~366`` apart, i.e. maximally scattered.

        The store is populated through the ``Dataset`` write API (``add_array`` /
        ``update_metadata``) rather than by hand-writing zarr, so the fixture
        depends on the same storage abstraction the production creator uses, not
        on the physical on-disk layout.
        """
        import datetime
        import os

        from anemoi.datasets.create.dataset import Dataset
        from anemoi.datasets.create.trajectories.creator import TrajectoryGriddedCreator

        full_bd, _ = self._make_recipe([]).make_groups().provider.factorise()
        keep = {datetime.datetime(y, 1, d) for y in (2020, 2021) for d in range(1, 6)}
        missing = [d.isoformat() for d in full_bd if d not in keep]

        recipe = self._make_recipe(missing)
        base_dates, steps = recipe.make_groups().provider.factorise()
        n, nsteps, nvars, ncells = len(base_dates), len(steps), 1, 2
        bd_np = np.array(base_dates, "datetime64[s]")

        data = np.arange(n * nvars * 1 * nsteps * ncells, dtype=np.float64).reshape(n, nvars, 1, nsteps, ncells)
        missing_idx = [i for i, d in enumerate(base_dates) if d not in keep]
        data[missing_idx] = np.nan

        zpath = os.path.join(str(tmp_path), "traj.zarr")
        ds = Dataset(zpath, create=True)
        ds.add_array(
            name="data",
            dimensions=("time", "variable", "ensemble", "step", "cell"),
            shape=data.shape,
            chunks=(1, nvars, 1, nsteps, ncells),
            dtype="float64",
            fill_value=np.nan,
        )
        ds.data[:] = data
        ds.add_array(name="base_dates", dimensions=("time",), data=bd_np)
        ds.add_array(name="steps", dimensions=("step",), data=np.asarray(steps))
        ds.update_metadata(variables=["v0"], missing_dates=[str(bd_np[i]) for i in missing_idx])

        wd = os.path.join(str(tmp_path), "wd")
        creator = TrajectoryGriddedCreator(recipe=recipe, path=zpath, parts="1/1", work_dir=wd)
        return creator, zpath, wd, missing_idx

    def test_split_branch_matches_full_compute(self, tmp_path):
        import glob
        import os

        from anemoi.datasets.create.dataset import Dataset
        from anemoi.datasets.create.statistics import TrajectoryStatisticsCollector

        creator, zpath, wd, missing_idx = self._build(tmp_path)
        dataset = Dataset(zpath, update=True)

        # Sanity: groups are scattered along axis 0 (the root cause of the bug).
        assert len(creator.groups) == 5
        idx0 = creator._group_row_indices(dataset, 0)
        assert idx0.tolist() == [0, 366]  # 2020-01-01 and 2021-01-01, one year apart

        # No-parts branch: full-dataset compute (reads every row, NaN skipped).
        reference = creator._compute_partial_statistics(dataset, 0, dataset.data.shape[0])

        # Parts branch: real ``load_done`` per group writes one pickle each ...
        for g in range(len(creator.groups)):
            creator.load_done(dataset, g)
        paths = sorted(glob.glob(os.path.join(wd, "statistics_*.pkl")))
        assert len(paths) == 5
        # ... and finalisation merges them.
        merged = TrajectoryStatisticsCollector.load_precomputed(dataset, paths)

        got = merged.statistics()
        expected = reference.statistics()
        assert set(got) == set(expected)
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-9, atol=1e-9, err_msg=key)
        assert any(k.startswith("statistics_tendencies_6h_") for k in got)

    def test_no_parts_branch_writes_no_pickles(self, tmp_path):
        import glob
        import os

        from anemoi.datasets.create.dataset import Dataset

        creator, zpath, wd, _ = self._build(tmp_path)
        creator.parts = None  # single full-dataset load -> statistics at finalise only
        dataset = Dataset(zpath, update=True)

        for g in range(len(creator.groups)):
            creator.load_done(dataset, g)

        assert glob.glob(os.path.join(wd, "statistics_*.pkl")) == []


# ---------------------------------------------------------------------------
# Build side: gridded rejects non-contiguous MMDD grouping
# ---------------------------------------------------------------------------


class TestGriddedRejectsMMDD:

    def test_mmdd_raises(self):
        from anemoi.datasets.create.gridded.creator import SimpleGriddedCreator

        stub = SimpleNamespace(recipe=SimpleNamespace(build=SimpleNamespace(group_by="MMDD")))
        with pytest.raises(AssertionError, match="MMDD"):
            SimpleGriddedCreator.initialise_dataset(stub, dataset=None)

    def test_contiguous_grouping_passes_guard(self):
        # A date-ordered grouping gets past the guard; it then fails on the stub
        # (no ``groups`` attribute), which confirms the assert itself did not fire.
        from anemoi.datasets.create.gridded.creator import SimpleGriddedCreator

        stub = SimpleNamespace(recipe=SimpleNamespace(build=SimpleNamespace(group_by="monthly")))
        with pytest.raises(Exception) as exc:
            SimpleGriddedCreator.initialise_dataset(stub, dataset=None)
        assert not isinstance(exc.value, AssertionError)
