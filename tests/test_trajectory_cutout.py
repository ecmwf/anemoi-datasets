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
