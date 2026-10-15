# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Statistics computed by ``finalise --load --parts i/n`` over contiguous row ranges.

The reference is a single pass over ``[0, N)``. The parts, each primed with the rows that
precede them (tendency warm-up) and merged by ``load_precomputed``, must give the same result.
"""

import glob
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from anemoi.datasets.create.gridded.creator import SimpleGriddedCreator
from anemoi.datasets.create.parts import part_bounds
from anemoi.datasets.create.recipe.statistics import StatisticsFilter
from anemoi.datasets.create.statistics import StatisticsCollector

C = 3  # variables
G = 4  # grid points


class _Dataset:
    """Minimal stand-in for :class:`anemoi.datasets.create.dataset.Dataset`."""

    def __init__(self, data: np.ndarray, chunk_rows: int, ready: bool = True) -> None:
        self.data = zarr.array(data, chunks=(chunk_rows,) + data.shape[1:])
        start = np.datetime64("2020-01-01T00:00:00", "s")
        self.dates = start + np.arange(data.shape[0]) * np.timedelta64(1, "h")
        self.store = {"_build": None}
        self._ready = ready

    def ready(self) -> bool:
        return self._ready

    def todo_remaining(self) -> int:
        return 0 if self._ready else 1


def _make_data(n: int, nan_rows: list[int] = ()) -> np.ndarray:
    rng = np.random.default_rng(42)
    # A trend makes tendencies non-trivial, the noise makes every row distinct.
    data = np.arange(n)[:, None, None, None] * np.linspace(0.5, 2, C)[None, :, None, None] + rng.normal(
        size=(n, C, 1, G)
    )
    data[:, 0] = 7.0  # a constant variable
    data = data.astype(np.float32)
    for i in nan_rows:
        data[i, 1:] = np.nan  # missing date (constant variable kept, as in a real dataset)
    return data


def _creator(work_dir: str, tendencies: dict[str, int], start=None, end=None, parts=None) -> SimpleGriddedCreator:
    creator = SimpleGriddedCreator.__new__(SimpleGriddedCreator)
    creator.path = os.path.join(work_dir, "dataset.zarr")
    creator.work_dir = os.path.join(work_dir, "dataset.zarr.work_dir")
    creator.parts = parts
    creator.__dict__["variables_names"] = [f"v{i}" for i in range(C)]

    def statistics_filter(dates):
        return StatisticsFilter(
            dates[0] if start is None else dates[start],
            dates[-1] if end is None else dates[end],
        )

    creator.recipe = SimpleNamespace(statistics=SimpleNamespace(statistics_filter=statistics_filter))
    creator._tendencies_to_compute = lambda dataset: tendencies
    return creator


def _run_parts(tmp_path, dataset, n, **kwargs) -> StatisticsCollector:
    creator = _creator(str(tmp_path), **kwargs)
    creator.finalise_prepare(dataset)
    for i in range(n):
        creator.parts = [f"{i+1}/{n}"]
        creator.finalise_load(dataset)
    return StatisticsCollector.load_precomputed(dataset, creator._statistics_part_files())


def _reference(tmp_path, dataset, **kwargs) -> StatisticsCollector:
    creator = _creator(str(tmp_path), **kwargs)
    return creator._compute_partial_statistics(dataset, 0, dataset.data.shape[0])


def _expected_tendencies(data, delta, start, end):
    """Brute force: tendencies at t with both t and t - delta in [start, end]."""
    sub = data[start : end + 1].astype(np.float64)
    if len(sub) <= delta:
        return None
    diff = (data[start + delta : end + 1] - data[start : end + 1 - delta]).astype(np.float64)
    out = {}
    for c in range(C):
        values = diff[:, c].ravel()
        values = values[~np.isnan(values)]
        out.setdefault("mean", []).append(values.mean())
        out.setdefault("minimum", []).append(values.min())
        out.setdefault("maximum", []).append(values.max())
        out.setdefault("stdev", []).append(values.std())
    return {k: np.array(v) for k, v in out.items()}


def _compare(actual: StatisticsCollector, expected: StatisticsCollector) -> None:
    a, e = actual.statistics(), expected.statistics()
    assert a.keys() == e.keys()
    for name in e:
        if "minimum" in name or "maximum" in name:
            np.testing.assert_array_equal(a[name], e[name], err_msg=name)
        else:
            np.testing.assert_allclose(a[name], e[name], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=name)
    assert actual.constant_variables() == expected.constant_variables()


TENDENCIES = [
    pytest.param({"1h": 1}, id="d1"),
    pytest.param({"1h": 1, "3h": 3, "6h": 6, "12h": 12, "24h": 24}, id="d1-24"),
    pytest.param({"6h": 6, "24h": 24}, id="d6-24"),
]


@pytest.mark.parametrize("tendencies", TENDENCIES)
@pytest.mark.parametrize("chunk_rows", [1, 5, 24])
@pytest.mark.parametrize("n_parts", [1, 2, 3, 7, "N", "N+3"])
def test_parts_equivalence(tmp_path, tendencies, chunk_rows, n_parts):
    N = 100
    dataset = _Dataset(_make_data(N), chunk_rows)
    nchunks = -(-N // chunk_rows)
    n = {"N": nchunks, "N+3": nchunks + 3}.get(n_parts, n_parts)

    merged = _run_parts(tmp_path, dataset, n, tendencies=tendencies)
    reference = _reference(tmp_path, dataset, tendencies=tendencies)
    _compare(merged, reference)

    stats = reference.statistics()
    for name, delta in tendencies.items():
        expected = _expected_tendencies(dataset.data[:], delta, 0, N - 1)
        for key, value in expected.items():
            np.testing.assert_allclose(stats[f"statistics_tendencies_{name}_{key}"], value, rtol=1e-9)


@pytest.mark.parametrize("tendencies", TENDENCIES)
@pytest.mark.parametrize(
    "start,end",
    [
        (None, None),
        (3, None),  # start inside the warm-up of the second part
        (20, None),  # start exactly at a part boundary
        (21, 78),  # start/end inside a part
        (35, 41),  # whole period inside one part, shorter than the largest delta
        (39, 79),  # end exactly before a part boundary
        (None, 40),  # end exactly at a part boundary
        (90, None),
    ],
)
def test_parts_filter_at_boundaries(tmp_path, tendencies, start, end):
    N = 100
    dataset = _Dataset(_make_data(N), chunk_rows=10)
    merged = _run_parts(tmp_path, dataset, 5, tendencies=tendencies, start=start, end=end)
    reference = _reference(tmp_path, dataset, tendencies=tendencies, start=start, end=end)
    _compare(merged, reference)

    stats = reference.statistics()
    for name, delta in tendencies.items():
        expected = _expected_tendencies(dataset.data[:], delta, start or 0, N - 1 if end is None else end)
        key = f"statistics_tendencies_{name}_mean"
        if expected is None:
            assert np.all(np.isnan(stats[key][1:])), (name, stats[key])
            continue
        for k, value in expected.items():
            np.testing.assert_allclose(stats[f"statistics_tendencies_{name}_{k}"], value, rtol=1e-9)


@pytest.mark.parametrize("nan_rows", [[19, 20], [17, 18, 19], [0, 1, 40], [55]])
def test_parts_missing_dates(tmp_path, nan_rows):
    N = 100
    tendencies = {"1h": 1, "3h": 3, "6h": 6}
    dataset = _Dataset(_make_data(N, nan_rows), chunk_rows=10)
    merged = _run_parts(tmp_path, dataset, 5, tendencies=tendencies)
    reference = _reference(tmp_path, dataset, tendencies=tendencies)
    _compare(merged, reference)

    stats = reference.statistics()
    for name, delta in tendencies.items():
        expected = _expected_tendencies(dataset.data[:], delta, 0, N - 1)
        for k, value in expected.items():
            np.testing.assert_allclose(stats[f"statistics_tendencies_{name}_{k}"], value, rtol=1e-9)


def test_parts_restart(tmp_path):
    dataset = _Dataset(_make_data(60), chunk_rows=10)
    tendencies = {"1h": 1, "6h": 6}
    creator = _creator(str(tmp_path), tendencies=tendencies)
    creator.finalise_prepare(dataset)

    creator.parts = ["1/3"]
    creator.finalise_load(dataset)
    (path,) = creator._statistics_part_files()
    mtime = os.stat(path).st_mtime_ns

    # Running the same part again is a no-op.
    creator.finalise_load(dataset)
    assert os.stat(path).st_mtime_ns == mtime

    # Killed between the write of the .tmp file and os.replace: the .tmp file is ignored.
    creator.parts = ["2/3"]
    lo, hi = 20, 40
    with open(creator._statistics_part_path(1, lo, hi) + ".tmp", "wb") as f:
        f.write(b"garbage")
    assert len(creator._statistics_part_files()) == 1
    creator.finalise_load(dataset)

    creator.parts = ["3/3"]
    creator.finalise_load(dataset)

    merged = StatisticsCollector.load_precomputed(dataset, creator._statistics_part_files())
    _compare(merged, _reference(tmp_path, dataset, tendencies=tendencies))


def test_parts_stale_and_missing(tmp_path):
    dataset = _Dataset(_make_data(60), chunk_rows=10)
    tendencies = {"1h": 1}

    # A previous run with 4 parts.
    _run_parts(tmp_path, dataset, 4, tendencies=tendencies)
    creator = _creator(str(tmp_path), tendencies=tendencies)
    assert len(creator._statistics_part_files()) == 4

    # --prepare removes the stale files.
    creator.finalise_prepare(dataset)
    assert creator._statistics_part_files() == []

    # A missing part makes the merge fail.
    for i in (1, 3):
        creator.parts = [f"{i}/3"]
        creator.finalise_load(dataset)
    with pytest.raises(ValueError, match="Missing statistics for group 1"):
        StatisticsCollector.load_precomputed(dataset, creator._statistics_part_files())


def test_parts_old_loader_pickles_ignored(tmp_path):
    dataset = _Dataset(_make_data(60), chunk_rows=10)
    tendencies = {"1h": 1}
    creator = _creator(str(tmp_path), tendencies=tendencies)
    os.makedirs(creator.work_dir)

    # Pickles written by the loaders of older versions.
    for group, (lo, hi) in enumerate([(0, 30), (30, 60)]):
        with open(os.path.join(creator.work_dir, f"statistics_{group:06d}-{lo:09d}-{hi:09d}.pkl"), "wb") as f:
            pickle.dump("old", f)

    assert creator._statistics_part_files() == []

    merged = _run_parts(tmp_path, dataset, 2, tendencies=tendencies)
    _compare(merged, _reference(tmp_path, dataset, tendencies=tendencies))
    assert len(glob.glob(os.path.join(creator.work_dir, "statistics_0*.pkl"))) == 2


def test_parts_prepare_requires_loaded_dataset(tmp_path):
    dataset = _Dataset(_make_data(20), chunk_rows=10, ready=False)
    creator = _creator(str(tmp_path), tendencies={})
    with pytest.raises(RuntimeError, match="not fully loaded"):
        creator.finalise_prepare(dataset)


@pytest.mark.parametrize("total", [0, 1, 5, 10])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 12])
def test_part_bounds_contiguous(total, n):
    bounds = [part_bounds(f"{i+1}/{n}", total) for i in range(n)]
    assert [b[0] for b in bounds] == list(range(n))
    assert bounds[0][1] == 0
    assert bounds[-1][2] == total
    for (_, _, hi), (_, lo, _) in zip(bounds[:-1], bounds[1:]):
        assert hi == lo
    assert part_bounds(None, total) == (0, 0, total)
    assert part_bounds(["all"], total) == (0, 0, total)
