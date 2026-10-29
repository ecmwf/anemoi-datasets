# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Regression tests for cutout with 5D trajectory datasets."""

from types import SimpleNamespace

import numpy as np
import pytest

from anemoi.datasets.usage.gridded.grids import Cutout


class ArrayDataset:
    """Minimal 5D dataset recording spatial read requests."""

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.grid_requests = []

    def __getitem__(self, index):
        self.grid_requests.append(index[-1])
        return self.data[index]


def make_trajectory_cutout():
    """Build a cutout with two regional grids and one global grid."""

    def dataset(ncells, offset):
        shape = (4, 2, 1, 3, ncells)
        data = np.arange(np.prod(shape), dtype=np.float32)
        return ArrayDataset(data.reshape(shape) + offset)

    lam1 = dataset(6, 1000)
    lam2 = dataset(5, 2000)
    globe = dataset(8, 3000)

    obj = object.__new__(Cutout)
    obj.axis = 4
    obj.lams = [lam1, lam2]
    obj.globe = globe
    obj.datasets = [lam1, lam2, globe]
    obj.forward = lam1

    obj.masks = [
        np.array([True, False, True, False, True, True]),
        np.array([True, True, False, True, False]),
    ]
    obj.global_mask = np.array([False, True, False, True, True, False, False, True])

    masks = [*obj.masks, obj.global_mask]

    reference = np.concatenate(
        [d.data[..., mask] for d, mask in zip(obj.datasets, masks)],
        axis=-1,
    )

    return obj, reference


@pytest.mark.parametrize(
    "index",
    [
        slice(None),
        1,
        (slice(None), slice(None), slice(None), slice(None), slice(None)),
        (slice(1, 4), slice(None), slice(None), slice(1, 3), slice(2, 9)),
        (slice(None), slice(None), slice(None), slice(None), slice(1, 11, 2)),
        (slice(None), slice(None), slice(None), slice(None), slice(7, 11)),
    ],
)
def test_trajectory_cutout_indexing(index):
    cutout, reference = make_trajectory_cutout()

    assert cutout.shape == (4, 2, 1, 3, 11)
    np.testing.assert_array_equal(cutout[index], reference[index])


def test_trajectory_cutout_global_only():
    cutout, reference = make_trajectory_cutout()

    index = (
        slice(None),
        slice(None),
        slice(None),
        slice(None),
        slice(7, 11),
    )

    np.testing.assert_array_equal(cutout[index], reference[index])

    assert cutout.lams[0].grid_requests == []
    assert cutout.lams[1].grid_requests == []
    assert len(cutout.globe.grid_requests) == 1


def temporal_dataset():
    return SimpleNamespace(
        shape=(4, 2, 1, 3, 6),
        base_dates=np.array(
            ["2025-01-01", "2025-01-02"],
            dtype="datetime64[h]",
        ),
        steps=np.array([0, 1, 2], dtype="timedelta64[h]"),
        base_frequency="24h",
        step_frequency="1h",
    )


def test_trajectory_cutout_rejects_different_base_dates():
    cutout, _ = make_trajectory_cutout()
    d1 = temporal_dataset()
    d2 = temporal_dataset()

    d2.base_dates = d2.base_dates + np.timedelta64(6, "h")

    with pytest.raises(ValueError, match="Incompatible base dates"):
        cutout.check_same_dates(d1, d2)


def test_trajectory_cutout_rejects_different_steps():
    cutout, _ = make_trajectory_cutout()
    d1 = temporal_dataset()
    d2 = temporal_dataset()

    d2.steps = np.array([0, 3, 6], dtype="timedelta64[h]")

    with pytest.raises(ValueError, match="Incompatible forecast steps"):
        cutout.check_same_dates(d1, d2)


def test_trajectory_cutout_rejects_different_frequencies():
    cutout, _ = make_trajectory_cutout()
    d1 = temporal_dataset()
    d2 = temporal_dataset()

    d2.step_frequency = "3h"

    with pytest.raises(ValueError, match="Incompatible step frequencies"):
        cutout.check_same_frequency(d1, d2)


def test_trajectory_cutout_real_constructor(monkeypatch):
    """Build a real Cutout with compatible synthetic 5D datasets."""
    from unittest.mock import MagicMock

    from anemoi.transform import spatial

    shape_lam = (2, 1, 1, 2, 3)
    shape_global = (2, 1, 1, 2, 4)

    lam_values = np.arange(np.prod(shape_lam), dtype=np.float32).reshape(shape_lam)

    global_values = np.arange(np.prod(shape_global), dtype=np.float32).reshape(shape_global) + 1000

    def make_source(data, latitudes, longitudes):
        ds = MagicMock()
        ds.shape = data.shape
        ds.sub_shape.side_effect = lambda axis: data.shape[:axis] + data.shape[axis + 1 :]
        ds.dtype = data.dtype
        ds.__len__.return_value = data.shape[0]
        ds._len = data.shape[0]
        ds.__getitem__.side_effect = data.__getitem__

        ds.latitudes = np.asarray(latitudes, dtype=float)
        ds.longitudes = np.asarray(longitudes, dtype=float)

        ds.variables = ["t"]
        ds.name_to_index = {"t": 0}
        ds.resolution = "synthetic"

        variable = MagicMock()
        variable.compatible.return_value = (True, None)
        ds.typed_variables = {"t": variable}

        ds.base_dates = np.array(
            ["2025-01-01", "2025-01-02"],
            dtype="datetime64[h]",
        )
        ds.steps = np.array([0, 1], dtype="timedelta64[h]")
        ds.base_frequency = "24h"
        ds.step_frequency = "1h"
        ds.frequency = "24h"
        ds.missing = set()

        return ds

    lam = make_source(
        lam_values,
        [40.0, 40.5, 41.0],
        [-5.0, -4.5, -4.0],
    )

    globe = make_source(
        global_values,
        [30.0, 40.0, 41.0, 50.0],
        [-10.0, -5.0, -4.0, 5.0],
    )

    def fake_cutout_mask(lam_lats, lam_lons, global_lats, global_lons, **kwargs):
        assert len(lam_lats) == 3
        assert len(global_lats) == 4
        return np.array([True, False, True, True])

    monkeypatch.setattr(spatial, "cutout_mask", fake_cutout_mask)

    # This exercises the real constructor and compatibility checks.
    combined = Cutout(
        [lam, globe],
        options={},
        neighbours=1,
    )

    assert combined.axis == 4

    expected = np.concatenate(
        [lam_values, global_values[..., [0, 2, 3]]],
        axis=-1,
    )

    assert combined.shape == expected.shape

    index = (slice(None),) * 5
    np.testing.assert_array_equal(combined[index], expected)

    np.testing.assert_array_equal(combined[0], expected[0])


class StepDataset:
    """Minimal trajectory dataset for automatic step alignment tests."""

    def __init__(self, hours):
        self.steps = np.array(hours, dtype="timedelta64[h]")
        self.base_dates = np.array(
            ["2025-01-01", "2025-01-02"],
            dtype="datetime64[h]",
        )
        self.variables = ["t"]
        self.shape = (2, 1, 1, len(hours), 4)
        self.data = np.arange(np.prod(self.shape)).reshape(self.shape)

    def mutate(self):
        return self

    def swap_with_parent(self, parent):
        return parent

    @property
    def step_frequency(self):
        from datetime import timedelta

        diffs = np.diff(self.steps)
        if len(diffs) == 0 or not np.all(diffs == diffs[0]):
            return None
        seconds = int(diffs[0] / np.timedelta64(1, "s"))
        return timedelta(seconds=seconds)

    def _subset(self, **kwargs):
        assert not kwargs
        return self

    def __getitem__(self, index):
        return self.data[index]


@pytest.mark.parametrize("reverse", [False, True])
def test_trajectory_cutout_adjusts_common_steps(reverse):
    from datetime import timedelta

    from anemoi.datasets.usage.misc import _auto_adjust

    hourly = StepDataset([6, 7, 8, 9, 10, 11, 12])
    three_hourly = StepDataset([6, 9, 12])

    sources = [hourly, three_hourly]
    if reverse:
        sources.reverse()

    adjusted, _ = _auto_adjust(sources, {"adjust": "all"})

    expected_steps = np.array([6, 9, 12], dtype="timedelta64[h]")

    for source, result in zip(sources, adjusted):
        np.testing.assert_array_equal(result.steps, expected_steps)
        assert result.shape == (2, 1, 1, 3, 4)
        assert result.step_frequency == timedelta(hours=3)

        indices = np.flatnonzero(np.isin(source.steps, expected_steps))
        np.testing.assert_array_equal(
            result[0],
            source.data[0][..., indices, :],
        )


def test_trajectory_cutout_adjust_keeps_matching_steps():
    from anemoi.datasets.usage.misc import _auto_adjust

    d1 = StepDataset([6, 9, 12])
    d2 = StepDataset([6, 9, 12])

    adjusted, _ = _auto_adjust([d1, d2], {"adjust": "all"})

    assert adjusted[0] is d1
    assert adjusted[1] is d2


@pytest.mark.parametrize(
    "other_steps",
    [
        [15, 18, 21],
        [6, 8, 11],
    ],
)
def test_trajectory_cutout_adjust_rejects_incompatible_steps(other_steps):
    from anemoi.datasets.usage.misc import _auto_adjust

    d1 = StepDataset([6, 7, 8, 9, 10, 11, 12])
    d2 = StepDataset(other_steps)

    with pytest.raises(ValueError, match="No regularly spaced common trajectory steps"):
        _auto_adjust([d1, d2], {"adjust": "all"})


def test_trajectory_cutout_adjust_select_does_not_change_steps():
    from anemoi.datasets.usage.misc import _auto_adjust

    d1 = StepDataset([6, 7, 8, 9, 10, 11, 12])
    d2 = StepDataset([6, 9, 12])

    adjusted, _ = _auto_adjust([d1, d2], {"adjust": "select"})

    assert adjusted[0] is d1
    assert adjusted[1] is d2


def test_trajectory_cutout_stepsubset_tendencies():
    from datetime import timedelta
    from unittest.mock import MagicMock

    from anemoi.datasets.usage.trajectories.subset import StepSubset

    source = MagicMock()
    source.mutate.return_value = source
    source.steps = np.arange(7, dtype="timedelta64[h]")

    subset = StepSubset(source, [0, 3, 6])

    subset.statistics_tendencies()
    source.statistics_tendencies.assert_called_once_with(timedelta(hours=3))

    source.statistics_tendencies.reset_mock()

    subset.statistics_tendencies(timedelta(hours=1))
    source.statistics_tendencies.assert_called_once_with(timedelta(hours=1))


def test_trajectory_cutout_stepsubset_irregular_tendencies():
    from unittest.mock import MagicMock

    from anemoi.datasets.usage.trajectories.subset import StepSubset

    source = MagicMock()
    source.mutate.return_value = source
    source.steps = np.arange(7, dtype="timedelta64[h]")

    subset = StepSubset(source, [0, 2, 5])

    assert subset.step_frequency is None

    with pytest.raises(ValueError, match="not uniformly spaced"):
        subset.statistics_tendencies()

    source.statistics_tendencies.assert_not_called()


def test_trajectory_cutout_adjust_two_and_five_hourly():
    from datetime import timedelta

    from anemoi.datasets.usage.misc import _auto_adjust

    two_hourly = StepDataset(list(range(0, 31, 2)))
    five_hourly = StepDataset(list(range(0, 31, 5)))

    sources = [two_hourly, five_hourly]
    adjusted, _ = _auto_adjust(sources, {"adjust": "all"})

    expected = np.array([0, 10, 20, 30], dtype="timedelta64[h]")

    for source, result in zip(sources, adjusted):
        np.testing.assert_array_equal(result.steps, expected)
        assert result.step_frequency == timedelta(hours=10)
        assert result.shape == (2, 1, 1, 4, 4)

        indices = np.flatnonzero(np.isin(source.steps, expected))
        np.testing.assert_array_equal(
            result[0],
            source.data[0][..., indices, :],
        )


@pytest.mark.parametrize(
    "first_has_stats,second_has_stats",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_trajectory_cutout_tendencies_follow_first_dataset(first_has_stats, second_has_stats):
    from datetime import timedelta
    from unittest.mock import MagicMock

    from anemoi.datasets.usage.forwards import Combined
    from anemoi.datasets.usage.gridded.grids import Cutout

    first = MagicMock()
    second = MagicMock()
    first.mutate.return_value = first

    delta = timedelta(hours=10)
    first_stats = {"stdev": np.array([2.0])}
    second_stats = {"stdev": np.array([3.0])}

    if first_has_stats:
        first.statistics_tendencies.return_value = first_stats
    else:
        first.statistics_tendencies.side_effect = KeyError("10h")

    if second_has_stats:
        second.statistics_tendencies.return_value = second_stats
    else:
        second.statistics_tendencies.side_effect = KeyError("10h")

    # Initialise the existing Combined machinery without spatial masks.
    cutout = Cutout.__new__(Cutout)
    cutout.check_compatibility = MagicMock()
    Combined.__init__(cutout, [first, second], MagicMock())

    if first_has_stats:
        assert cutout.statistics_tendencies(delta) is first_stats
    else:
        with pytest.raises(KeyError):
            cutout.statistics_tendencies(delta)

    first.statistics_tendencies.assert_called_once_with(delta)
    second.statistics_tendencies.assert_not_called()


@pytest.mark.parametrize("reverse", [False, True])
def test_trajectory_cutout_adjust_offset_base_dates_and_steps(reverse):
    from datetime import timedelta

    from anemoi.datasets.usage.misc import _auto_adjust

    origin = np.datetime64("2025-01-01T00:00", "h")

    regional = StepDataset(list(range(0, 31, 2)))
    global_ds = StepDataset(list(range(0, 31, 3)))

    regional.base_dates = origin + np.arange(22) * np.timedelta64(7, "h")
    global_ds.base_dates = origin + np.timedelta64(11, "h") + np.arange(30) * np.timedelta64(5, "h")

    regional.base_frequency = timedelta(hours=7)
    global_ds.base_frequency = timedelta(hours=5)

    for ds in (regional, global_ds):
        ds.shape = (len(ds.base_dates), 1, 1, len(ds.steps), 4)
        ds.data = np.arange(np.prod(ds.shape)).reshape(ds.shape)

    sources = [regional, global_ds]
    if reverse:
        sources.reverse()

    adjusted, _ = _auto_adjust(sources, {"adjust": "all"})

    expected_base_dates = origin + np.array([21, 56, 91, 126], dtype="timedelta64[h]")
    expected_steps = np.array([0, 6, 12, 18, 24, 30], dtype="timedelta64[h]")

    for source, result in zip(sources, adjusted):
        np.testing.assert_array_equal(result.base_dates, expected_base_dates)
        np.testing.assert_array_equal(result.steps, expected_steps)

        assert result.base_frequency == timedelta(hours=35)
        assert result.step_frequency == timedelta(hours=6)
        assert result.shape == (4, 1, 1, 6, 4)

        base_indices = np.flatnonzero(np.isin(source.base_dates, expected_base_dates))
        step_indices = np.flatnonzero(np.isin(source.steps, expected_steps))

        for i, j in enumerate(base_indices):
            np.testing.assert_array_equal(
                result[i],
                source.data[j][..., step_indices, :],
            )


@pytest.mark.parametrize("reverse", [False, True])
def test_trajectory_cutout_adjust_partial_time_overlap(reverse):
    from datetime import timedelta

    from anemoi.datasets.usage.misc import _auto_adjust

    origin = np.datetime64("2025-01-01T00:00", "h")

    regional = StepDataset(list(range(2, 29, 2)))
    global_ds = StepDataset(list(range(6, 25, 3)))

    regional.base_dates = origin + np.arange(15) * np.timedelta64(7, "h")
    global_ds.base_dates = origin + np.timedelta64(11, "h") + np.arange(18) * np.timedelta64(5, "h")

    regional.base_frequency = timedelta(hours=7)
    global_ds.base_frequency = timedelta(hours=5)

    for ds in (regional, global_ds):
        ds.shape = (len(ds.base_dates), 1, 1, len(ds.steps), 4)
        ds.data = np.arange(np.prod(ds.shape)).reshape(ds.shape)

    sources = [regional, global_ds]
    if reverse:
        sources.reverse()

    adjusted, _ = _auto_adjust(sources, {"adjust": "all"})

    expected_dates = origin + np.array([21, 56, 91], dtype="timedelta64[h]")
    expected_steps = np.array([6, 12, 18, 24], dtype="timedelta64[h]")

    for source, result in zip(sources, adjusted):
        np.testing.assert_array_equal(result.base_dates, expected_dates)
        np.testing.assert_array_equal(result.steps, expected_steps)

        assert result.base_frequency == timedelta(hours=35)
        assert result.step_frequency == timedelta(hours=6)
        assert result.shape == (3, 1, 1, 4, 4)

        assert result.base_dates[0] >= source.base_dates[0]
        assert result.base_dates[-1] <= source.base_dates[-1]
        assert result.steps[0] >= source.steps[0]
        assert result.steps[-1] <= source.steps[-1]

        base_indices = np.flatnonzero(np.isin(source.base_dates, expected_dates))
        step_indices = np.flatnonzero(np.isin(source.steps, expected_steps))

        for i, j in enumerate(base_indices):
            np.testing.assert_array_equal(
                result[i],
                source.data[j][..., step_indices, :],
            )


def test_trajectory_cutout_adjust_no_common_base_dates():
    from anemoi.datasets.usage.misc import _auto_adjust

    origin = np.datetime64("2025-01-01T00:00", "h")

    regional = StepDataset([0, 2, 4, 6])
    global_ds = StepDataset([0, 2, 4, 6])

    regional.base_dates = origin + np.arange(4) * np.timedelta64(7, "h")
    global_ds.base_dates = origin + np.timedelta64(35, "h") + np.arange(4) * np.timedelta64(5, "h")

    with pytest.raises(ValueError, match="No common trajectory base dates"):
        _auto_adjust([regional, global_ds], {"adjust": "all"})
