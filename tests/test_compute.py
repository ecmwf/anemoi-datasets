# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``anemoi-datasets compute`` command.

These exercise the standalone statistics/tendency/residual engine using the
on-the-fly fake datasets defined in :mod:`test_data` (names such as
``test-2021-2021-6h-o96-abcd-0``). Two such datasets that differ only by their
``k`` index differ by a constant everywhere, which makes residual statistics easy
to check exactly.
"""

import argparse
import json

import numpy as np
import pytest
from test_data import mockup_open_zarr

from anemoi.datasets import open_dataset
from anemoi.datasets.commands.compute import Compute
from anemoi.datasets.commands.compute import _parse
from anemoi.datasets.commands.compute.engine import Collectors
from anemoi.datasets.commands.compute.engine import Task
from anemoi.datasets.commands.compute.engine import _blocks
from anemoi.datasets.commands.compute.engine import _read_block
from anemoi.datasets.commands.compute.engine import _render_live
from anemoi.datasets.commands.compute.engine import _sample_indices
from anemoi.datasets.commands.compute.engine import _save_checkpoint
from anemoi.datasets.commands.compute.engine import _seed_from_sha
from anemoi.datasets.commands.compute.engine import run as run_engine
from anemoi.datasets.commands.compute.statistics import Accumulator
from anemoi.datasets.commands.compute.statistics_tendencies import TendencyAccumulator

# A small fake dataset: 1 month at 6h, 4 variables, grid of 10 points.
DS0 = "test-2021-2021-6h-o96-abcd-0"
DS1 = "test-2021-2021-6h-o96-abcd-1"


def _expected_statistics(name: str) -> dict[str, np.ndarray]:
    """Reference statistics computed directly from the dataset data with numpy."""
    data = np.asarray(open_dataset(name)[:], dtype=np.float64)  # (T, V, E, G)
    flat = np.moveaxis(data, 1, 0).reshape(data.shape[1], -1)
    return {
        "mean": np.nanmean(flat, axis=1),
        "stdev": np.nanstd(flat, axis=1),
        "minimum": np.nanmin(flat, axis=1),
        "maximum": np.nanmax(flat, axis=1),
    }


def _make_task(**kwargs) -> Task:
    base = dict(open_args=[DS0], open_kwargs={}, label=DS0, args_sha="t")
    base.update(kwargs)
    return Task(**base)


# --------------------------------------------------------------------------- #
# Pure parsing (no dataset needed)
# --------------------------------------------------------------------------- #


def test_parse_name_form() -> None:
    p = _parse([DS0, "start=2021", "--statistics", "--statistics-tendencies", "6h", "--parallel", "3"])
    assert p.open_args == [DS0]
    assert p.open_kwargs == {"start": 2021}
    assert p.do_statistics is True
    assert p.tendency == "6h"
    assert p.parallel == 3
    assert p.allow_nans is True  # NaNs ignored by default


def test_parse_json_form() -> None:
    cfg = json.dumps({"dataset": DS0, "start": "2021-01-01"})
    p = _parse([cfg, "--statistics", "--parallel", "8"])
    assert p.open_args == [{"dataset": DS0, "start": "2021-01-01"}]
    assert p.open_kwargs == {}
    assert p.parallel == 8


def test_parse_json_rejects_keyvalue() -> None:
    cfg = json.dumps({"dataset": DS0})
    with pytest.raises(ValueError, match="not allowed when the dataset is a JSON"):
        _parse([cfg, "extra=1"])


def test_parse_residual_with_trailing_flags() -> None:
    p = _parse([DS0, "--statistics-residual", DS1, "thinning=4", "--output", "x.json", "--parallel", "2"])
    assert p.has_residual is True
    assert p.residual_open_args == [DS1]
    assert p.residual_open_kwargs == {"thinning": 4}
    assert p.output == "x.json"
    assert p.parallel == 2


def test_parse_residual_json() -> None:
    cfg = json.dumps({"dataset": DS1, "grid": "o96"})
    p = _parse([DS0, "--statistics-residual", cfg, "--output", "y.json"])
    assert p.residual_open_args == [{"dataset": DS1, "grid": "o96"}]
    assert p.output == "y.json"


def test_parse_old_flags_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown option"):
        _parse([DS0, "--tendencies", "6h"])
    with pytest.raises(ValueError, match="Unknown option"):
        _parse([DS0, "--residual", DS1])


def test_default_action_is_statistics() -> None:
    assert _parse([DS0]).do_statistics is True


def test_parse_sample_dates() -> None:
    assert _parse([DS0, "--sample-dates", "0.1"]).sample_dates == 0.1


# --------------------------------------------------------------------------- #
# Engine: statistics / tendencies / residual on fake datasets
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_statistics_match_numpy() -> None:
    _, results = run_engine(_make_task(do_statistics=True))
    expected = _expected_statistics(DS0)
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(results["statistics"][key], expected[key])
    assert results["tendency"] is None


@mockup_open_zarr
def test_tendency_match_numpy() -> None:
    _, results = run_engine(_make_task(do_statistics=False, tendency="6h"))
    data = np.asarray(open_dataset(DS0)[:], dtype=np.float64)
    tend = data[1:] - data[:-1]  # 6h == one step
    flat = np.moveaxis(tend, 1, 0).reshape(tend.shape[1], -1)
    np.testing.assert_allclose(results["tendency"]["mean"], np.nanmean(flat, axis=1))
    np.testing.assert_allclose(results["tendency"]["stdev"], np.nanstd(flat, axis=1))
    assert results["statistics"] is None


@mockup_open_zarr
def test_statistics_parallel_equals_sequential() -> None:
    _, seq = run_engine(_make_task(do_statistics=True, tendency="6h", chunk_size=7))
    _, par = run_engine(_make_task(do_statistics=True, tendency="6h", chunk_size=7, parallel=4))
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(seq["statistics"][key], par["statistics"][key])
        np.testing.assert_allclose(seq["tendency"][key], par["tendency"][key])


@mockup_open_zarr
def test_residual_is_constant() -> None:
    # DS0 and DS1 differ only by k (0 vs 1), i.e. by 0.1 everywhere, so the
    # residual DS0 - DS1 is the constant -0.1 for every variable.
    task = _make_task(
        do_statistics=True,
        has_residual=True,
        residual_open_args=[DS1],
        residual_open_kwargs={},
        residual_label=DS1,
    )
    _, results = run_engine(task)
    n = len(open_dataset(DS0).variables)
    # The fake data has values around 2e9, so float64 rounding makes the residual
    # -0.1 to within a few 1e-7 rather than exactly; that is a property of the test
    # data, not of the accumulator.
    np.testing.assert_allclose(results["statistics"]["mean"], np.full(n, -0.1), atol=1e-5)
    np.testing.assert_allclose(results["statistics"]["stdev"], np.zeros(n), atol=1e-5)
    np.testing.assert_allclose(results["statistics"]["minimum"], np.full(n, -0.1), atol=1e-5)
    np.testing.assert_allclose(results["statistics"]["maximum"], np.full(n, -0.1), atol=1e-5)


@mockup_open_zarr
def test_checkpoint_resume_matches_full() -> None:
    _, full = run_engine(_make_task(do_statistics=True, tendency="6h", chunk_size=5, args_sha="res"))

    # Manually compute the first half of the blocks into a checkpoint, then resume.
    ds = open_dataset(DS0)
    n = len(ds)
    blocks = _blocks(n, 5, None)
    done = len(blocks) // 2
    col = Collectors(list(ds.variables), True, 1, True)
    for b in range(done):
        col.update(_read_block(ds, None, blocks[b]))
    ckpt = "/tmp/compute_test_resume.pkl"
    _save_checkpoint(
        ckpt,
        {
            "version": 1,
            "args_sha": "res",
            "mode": "sequential",
            "collectors": col,
            "next_block": done,
            "progress": f"{done}/{len(blocks)} blocks",
        },
    )
    _, resumed = run_engine(
        _make_task(do_statistics=True, tendency="6h", chunk_size=5, args_sha="res", checkpoint_path=ckpt, resume=True)
    )
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(full["statistics"][key], resumed["statistics"][key])
        np.testing.assert_allclose(full["tendency"][key], resumed["tendency"][key])


# --------------------------------------------------------------------------- #
# Date subsampling
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_sample_dates_matches_numpy_on_sample() -> None:
    frac = 0.3
    task = _make_task(do_statistics=True, sample_dates=frac, args_sha="smp")
    _, results = run_engine(task)

    ds = open_dataset(DS0)
    idx = _sample_indices(len(ds), frac, _seed_from_sha("smp"))
    data = np.asarray(ds[list(idx)], dtype=np.float64)
    flat = np.moveaxis(data, 1, 0).reshape(data.shape[1], -1)
    np.testing.assert_allclose(results["statistics"]["mean"], np.nanmean(flat, axis=1))
    np.testing.assert_allclose(results["statistics"]["stdev"], np.nanstd(flat, axis=1))


@mockup_open_zarr
def test_sample_dates_rejected_with_tendency() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        run_engine(_make_task(do_statistics=True, tendency="6h", sample_dates=0.5))


@mockup_open_zarr
def test_sample_dates_rejected_with_parallel() -> None:
    with pytest.raises(ValueError, match="not supported with --parallel"):
        run_engine(_make_task(do_statistics=True, sample_dates=0.5, parallel=4))


def test_sample_indices_is_deterministic() -> None:
    a = _sample_indices(1000, 0.1, 42)
    b = _sample_indices(1000, 0.1, 42)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 100
    assert (np.diff(a) > 0).all()  # sorted, unique


# --------------------------------------------------------------------------- #
# Live statistics table
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_live_render_does_not_raise(capsys) -> None:
    ds = open_dataset(DS0)
    col = Collectors(list(ds.variables), True, None, True)
    col.update(np.asarray(ds[0:4], dtype=np.float64))
    _render_live(col, list(ds.variables), [0, 1, 2])
    out = capsys.readouterr().out
    assert "Statistics" in out
    assert ds.variables[0] in out


@mockup_open_zarr
def test_run_with_live_enabled_smoke() -> None:
    _, results = run_engine(_make_task(do_statistics=True, live=True, chunk_size=200))
    assert results["statistics"] is not None


# --------------------------------------------------------------------------- #
# Full command (CLI entry) with JSON output
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_command_writes_json(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # keep the default checkpoint out of the repo
    out = tmp_path / "out.json"
    Compute().run(argparse.Namespace(rest=[DS0, "--statistics", "--output", str(out)]))
    doc = json.loads(out.read_text())
    assert doc["dataset"] == DS0
    assert doc["statistics"] is not None
    assert len(doc["statistics"]["mean"]) == len(open_dataset(DS0).variables)


@mockup_open_zarr
def test_command_json_equals_name_form(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    name_out = tmp_path / "name.json"
    json_out = tmp_path / "json.json"
    Compute().run(argparse.Namespace(rest=[DS0, "--statistics", "--output", str(name_out)]))
    cfg = json.dumps({"dataset": DS0})
    Compute().run(argparse.Namespace(rest=[cfg, "--statistics", "--output", str(json_out)]))
    a = json.loads(name_out.read_text())["statistics"]["mean"]
    b = json.loads(json_out.read_text())["statistics"]["mean"]
    np.testing.assert_allclose(a, b)


# --------------------------------------------------------------------------- #
# Accumulator units (no dataset)
# --------------------------------------------------------------------------- #


def test_accumulator_merge_equivalence() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 3, 2, 6))
    whole = Accumulator(["a", "b", "c"], allow_nans=True)
    whole.update(data)
    a = Accumulator(["a", "b", "c"], allow_nans=True)
    a.update(data[:17])
    b = Accumulator(["a", "b", "c"], allow_nans=True)
    b.update(data[17:])
    merged = a.merge(b)
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(whole.statistics()[key], merged.statistics()[key])


def test_accumulator_nan_policy() -> None:
    data = np.ones((4, 2, 1, 3))
    data[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN values found"):
        Accumulator(["a", "b"], allow_nans=False).update(data)
    # allow_nans=True ignores it
    Accumulator(["a", "b"], allow_nans=True).update(data)


def test_tendency_seed_window_equivalence() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(size=(30, 3, 1, 4))
    delta = 4
    full = TendencyAccumulator(["a", "b", "c"], delta, allow_nans=True)
    full.update(data)
    s1 = TendencyAccumulator(["a", "b", "c"], delta, allow_nans=True)
    s1.update(data[:18])
    s2 = TendencyAccumulator(["a", "b", "c"], delta, allow_nans=True)
    s2.seed_window(data[18 - delta : 18])
    s2.update(data[18:])
    merged = s1.merge(s2)
    np.testing.assert_allclose(full.statistics()["mean"], merged.statistics()["mean"])
    np.testing.assert_allclose(full.statistics()["stdev"], merged.statistics()["stdev"])
