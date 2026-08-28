# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for overlaying gridded datasets onto trajectories datasets.

``open_dataset(traj, gridded)`` (in any order, any number of each) broadcasts
every gridded dataset onto the trajectory layout by valid-time lookup
(``result[base=b, step=s] = gridded[b + s]``) and joins them along the variable
axis, preserving the argument order.  The result behaves like a trajectories
dataset.

Notes
-----

TO BE REFACTORED. These tests hand-build in-memory zarr groups
(``make_trajectories_zarr`` + a local ``_add_statistics`` workaround and a
local ``make_gridded_zarr``) because synthetic datasets do not yet support
the trajectories layout. Once synthetic datasets gain a trajectories layout,
the fixtures here should most likely be rebuilt on top of them, folding the
statistics/tendencies generation into the shared helpers and reusing the
canonical indexing sweep (``anemoi.datasets.misc.testing`` /
``default_test_indexing``) instead of the bespoke helpers below.
"""

import datetime

import numpy as np
import pytest
import zarr
from test_trajectories import make_trajectories_zarr

from anemoi.datasets import open_dataset
from anemoi.datasets.usage.store import ZarrStore

N_CELLS = 10


def make_gridded_zarr(
    n_dates: int,
    n_vars: int = 2,
    n_cells: int = N_CELLS,
    frequency_h: int = 6,
    start: datetime.datetime = datetime.datetime(2021, 1, 1),
    vars: list[str] | None = None,
    analytic: bool = True,
) -> zarr.Group:
    """Build a minimal in-memory zarr group with the gridded layout.

    The grid (``latitudes``/``longitudes``) matches
    :func:`test_trajectories.make_trajectories_zarr` so the two can be joined.

    Parameters
    ----------
    n_dates : int
        Number of dates.
    n_vars : int
        Number of variables.
    n_cells : int
        Number of grid cells.
    frequency_h : int
        Dataset frequency in hours.
    start : datetime.datetime
        First date.
    vars : list of str, optional
        Variable names.  Defaults to ``["g0", "g1", ...]``.
    analytic : bool
        When True, fill ``data`` with ``arange`` values so every element is
        unique and valid-time lookups can be checked exactly.

    Returns
    -------
    zarr.Group
        In-memory zarr group.
    """
    if vars is None:
        vars = [f"g{i}" for i in range(n_vars)]

    root = zarr.group()
    dates = np.array(
        [start + datetime.timedelta(hours=frequency_h * i) for i in range(n_dates)],
        dtype="datetime64[s]",
    )
    shape = (n_dates, n_vars, 1, n_cells)
    if analytic:
        data = np.arange(np.prod(shape), dtype="float64").reshape(shape)
    else:
        data = np.random.default_rng(1).random(shape)

    root.create_array("data", data=data, chunks=data.shape, compressors=None)
    root.create_array("dates", data=dates, compressors=None)
    root.create_array("latitudes", data=np.linspace(-90, 90, n_cells), compressors=None)
    root.create_array("longitudes", data=np.linspace(0, 360, n_cells), compressors=None)
    root.create_array("mean", data=np.mean(data, axis=(0, 2, 3)), compressors=None)
    root.create_array("stdev", data=np.std(data, axis=(0, 2, 3)), compressors=None)
    root.create_array("maximum", data=np.max(data, axis=(0, 2, 3)), compressors=None)
    root.create_array("minimum", data=np.min(data, axis=(0, 2, 3)), compressors=None)

    # Tendencies along the gridded time axis, keyed by the gridded frequency.
    tendencies = np.diff(data, axis=0) if data.shape[0] >= 2 else data
    for k, v in {
        "mean": np.mean(tendencies, axis=(0, 2, 3)),
        "stdev": np.std(tendencies, axis=(0, 2, 3)),
        "maximum": np.max(tendencies, axis=(0, 2, 3)),
        "minimum": np.min(tendencies, axis=(0, 2, 3)),
    }.items():
        root.create_array(f"statistics_tendencies_{frequency_h}h_{k}", data=v, compressors=None)

    root.attrs["layout"] = "gridded"
    root.attrs["frequency"] = f"{frequency_h}h"
    root.attrs["resolution"] = "test"
    root.attrs["name_to_index"] = {v: i for i, v in enumerate(vars)}
    root.attrs["variables_metadata"] = {v: {} for v in vars}
    root.attrs["data_request"] = {"grid": 1, "area": "g", "param_level": {}}
    return root


def _open(group: zarr.Group, path: str) -> ZarrStore:
    return ZarrStore.from_group(group, path=path)


def _add_statistics(group: zarr.Group, step_frequency_h: int = 6) -> zarr.Group:
    """Add per-variable statistics arrays (missing from the shared helper)."""
    data = group["data"][:]
    # Reduce over every axis except variables (axis 1) -> shape (n_vars,).
    axes = (0, 2, 3, 4)
    group.create_array("mean", data=np.mean(data, axis=axes), compressors=None)
    group.create_array("stdev", data=np.std(data, axis=axes), compressors=None)
    group.create_array("maximum", data=np.max(data, axis=axes), compressors=None)
    group.create_array("minimum", data=np.min(data, axis=axes), compressors=None)

    # Tendencies along the step axis (position -2), keyed by the step frequency.
    tendencies = np.diff(data, axis=-2)
    for k, v in {
        "mean": np.mean(tendencies, axis=axes),
        "stdev": np.std(tendencies, axis=axes),
        "maximum": np.max(tendencies, axis=axes),
        "minimum": np.min(tendencies, axis=axes),
    }.items():
        group.create_array(f"statistics_tendencies_{step_frequency_h}h_{k}", data=v, compressors=None)
    return group


@pytest.fixture
def traj():
    # base dates: 2021-01-01 00/06/12/18Z ; steps: 6..30h
    group = make_trajectories_zarr(n_dates=4, n_steps=5, n_vars=3, n_cells=N_CELLS, analytic=True)
    return _open(_add_statistics(group), "traj.zarr")


@pytest.fixture
def gridded():
    # 6h dataset covering the full valid-time envelope (up to 18Z + 30h = 48h)
    return _open(make_gridded_zarr(n_dates=9, n_vars=2), "grid.zarr")


def test_shape_and_variable_order(traj, gridded):
    """Trajectory-first and gridded-first give the expected shape and variable order."""
    ds = open_dataset(traj, gridded)
    # (base_dates, Vt + Vg, ensembles, steps, cells)
    assert ds.shape == (4, 5, 1, 5, N_CELLS)
    assert ds.variables == ["a", "b", "c", "g0", "g1"]

    reversed_ds = open_dataset(gridded, traj)
    assert reversed_ds.shape == (4, 5, 1, 5, N_CELLS)
    assert reversed_ds.variables == ["g0", "g1", "a", "b", "c"]


def test_behaves_like_trajectory(traj, gridded):
    """The result exposes the trajectory time axes, not a single ``dates`` axis."""
    ds = open_dataset(traj, gridded)
    assert ds.frequency is None
    assert np.array_equal(ds.base_dates, traj.base_dates)
    assert np.array_equal(ds.steps, traj.steps)
    assert ds.base_frequency == datetime.timedelta(hours=6)
    with pytest.raises(AttributeError):
        _ = ds.dates


def test_valid_time_broadcast_values(traj, gridded):
    """Each gridded slot equals the gridded field at valid time ``base + step``."""
    ds = open_dataset(traj, gridded)
    full = ds[:]

    gridded_dates = gridded.dates.astype("datetime64[s]")
    lookup = {d: i for i, d in enumerate(gridded_dates.tolist())}
    base_dates = traj.base_dates.astype("datetime64[s]")
    steps = traj.steps.astype("timedelta64[s]")
    gdata = gridded[:]
    n_traj_vars = len(traj.variables)

    for i in range(len(base_dates)):
        for j in range(len(steps)):
            k = lookup[(base_dates[i] + steps[j]).tolist()]
            # Trajectory variables are untouched; gridded variables broadcast.
            assert np.array_equal(full[i, :n_traj_vars, :, j, :], traj[i][:, :, j, :])
            assert np.array_equal(full[i, n_traj_vars:, :, j, :], gdata[k])


def test_duplication_on_shared_valid_time(traj, gridded):
    """The same gridded field is reused wherever two (base, step) pairs share a valid time."""
    ds = open_dataset(traj, gridded)
    full = ds[:]
    n_traj_vars = len(traj.variables)

    # base[0]=00Z, step 12h -> valid 12Z ; base[2]=12Z, step 0h is absent, but
    # base[1]=06Z + 6h -> 12Z and base[0]=00Z + 12h -> 12Z share valid time 12Z.
    assert np.array_equal(
        full[0, n_traj_vars:, :, 1, :],  # base 00Z, step 12h (index 1)
        full[1, n_traj_vars:, :, 0, :],  # base 06Z, step 6h  (index 0)
    )


@pytest.mark.parametrize(
    "index",
    [
        (1,),
        (1, slice(None), 0, 2, slice(None)),
        (slice(1, 3), slice(3, 5)),
        (slice(0, 2),),
    ],
)
def test_indexing_matches_full(traj, gridded, index):
    """Tuple/slice indexing matches indexing the materialised array."""
    ds = open_dataset(traj, gridded)
    full = ds[:]
    assert np.array_equal(ds[index], full[index])


def test_statistics_are_joined(traj, gridded):
    """Statistics are concatenated along the variable axis in argument order."""
    ds = open_dataset(traj, gridded)
    for key in ("mean", "stdev", "maximum", "minimum"):
        expected = np.concatenate([traj.statistics[key], gridded.statistics[key]], axis=0)
        assert np.array_equal(ds.statistics[key], expected)


def test_statistics_tendencies_forwarded(traj, gridded):
    """Tendencies do not raise; the gridded variables carry the gridded dataset's own values."""
    ds = open_dataset(traj, gridded)
    n_traj_vars = len(traj.variables)

    tendencies = ds.statistics_tendencies()  # must not raise
    for key in ("mean", "stdev", "maximum", "minimum"):
        # Trajectory part uses the step-frequency delta; gridded part is forwarded verbatim.
        assert np.array_equal(tendencies[key][:n_traj_vars], traj.statistics_tendencies()[key])
        assert np.array_equal(tendencies[key][n_traj_vars:], gridded.statistics_tendencies()[key])


def test_select_after_join(traj, gridded):
    """A ``select`` mixing trajectory and gridded variables keeps the trajectory layout."""
    ds = open_dataset(traj, gridded, select=["a", "g1"])
    assert ds.variables == ["a", "g1"]
    assert ds.shape == (4, 2, 1, 5, N_CELLS)


def test_step_subset_after_join(traj, gridded):
    """A ``steps`` selection narrows the step axis of the joined dataset."""
    ds = open_dataset(traj, gridded, steps=[6, 18])
    assert ds.shape[-2] == 2
    assert [s.astype("timedelta64[h]").astype(int) for s in ds.steps] == [6, 18]


def test_metadata_serialises(traj, gridded):
    """``metadata()`` is JSON-serialisable and carries the trajectory keys."""
    ds = open_dataset(traj, gridded)
    md = ds.metadata()
    assert md["frequency"] is None
    assert md["base_frequency"] == "6h"
    assert "specific" in md


def test_missing_valid_time_raises(traj):
    """A gridded dataset that does not cover every required valid time is rejected."""
    # Only three 6h dates -> covers 00/06/12Z, far short of the 48h envelope.
    short = _open(make_gridded_zarr(n_dates=3, n_vars=2), "short.zarr")
    with pytest.raises(ValueError, match="does not cover"):
        open_dataset(traj, short)


def test_grid_mismatch_raises(traj):
    """A gridded dataset on a different grid is rejected."""
    other_grid = _open(make_gridded_zarr(n_dates=9, n_vars=2, n_cells=N_CELLS + 1), "other.zarr")
    with pytest.raises(ValueError):
        open_dataset(traj, other_grid)


def test_multiple_gridded(traj):
    """Any number of gridded datasets can be overlaid onto a trajectory."""
    g1 = _open(make_gridded_zarr(n_dates=9, n_vars=1, vars=["x"]), "g1.zarr")
    g2 = _open(make_gridded_zarr(n_dates=9, n_vars=2, vars=["y", "z"]), "g2.zarr")
    ds = open_dataset(traj, g1, g2)
    assert ds.variables == ["a", "b", "c", "x", "y", "z"]
    assert ds.shape == (4, 6, 1, 5, N_CELLS)
