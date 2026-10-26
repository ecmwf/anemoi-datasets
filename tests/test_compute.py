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
from typing import Any

import numpy as np
import pytest
import zarr
from test_data import mockup_open_zarr

from anemoi.datasets import open_dataset
from anemoi.datasets.commands.compute import Compute
from anemoi.datasets.commands.compute import _default_statistics_output
from anemoi.datasets.commands.compute import _parse
from anemoi.datasets.commands.compute.engine import CHECKPOINT_VERSION
from anemoi.datasets.commands.compute.engine import Collectors
from anemoi.datasets.commands.compute.engine import Task
from anemoi.datasets.commands.compute.engine import _blocks
from anemoi.datasets.commands.compute.engine import _read_block
from anemoi.datasets.commands.compute.engine import _render_live
from anemoi.datasets.commands.compute.engine import _runs
from anemoi.datasets.commands.compute.engine import _sample_indices
from anemoi.datasets.commands.compute.engine import _save_checkpoint
from anemoi.datasets.commands.compute.engine import _seed_from_sha
from anemoi.datasets.commands.compute.engine import run as run_engine
from anemoi.datasets.commands.compute.interpolation import Interpolator
from anemoi.datasets.commands.compute.interpolation import TargetGrid
from anemoi.datasets.commands.compute.interpolation import _lookup_key
from anemoi.datasets.commands.compute.output_dataset import ConstantsTracker
from anemoi.datasets.commands.compute.output_dataset import OutputDataset
from anemoi.datasets.commands.compute.statistics import Accumulator
from anemoi.datasets.commands.compute.statistics_tendencies import TendencyAccumulator
from anemoi.datasets.misc.residual_statistics import load_document

# A small fake dataset: 1 month at 6h, 4 variables, grid of 10 points.
DS0 = "test-2021-2021-6h-o96-abcd-0"
DS1 = "test-2021-2021-6h-o96-abcd-1"
# Same dates and variables as DS0, but on a 5-point grid instead of a 10-point one.
DS_SMALL = "test-2021-2021-6h-o96-abcd-0--5-1,5"


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
    p = _parse([DS0, "--minus", DS1, "thinning=4", "--output-statistics", "x.npz", "--parallel", "2"])
    assert p.has_residual is True
    assert p.residual_open_args == [DS1]
    assert p.residual_open_kwargs == {"thinning": 4}
    assert p.output_statistics == "x.npz"
    assert p.parallel == 2


def test_parse_residual_json() -> None:
    cfg = json.dumps({"dataset": DS1, "grid": "o96"})
    p = _parse([DS0, "--minus", cfg, "--output-statistics", "y.npz"])
    assert p.residual_open_args == [{"dataset": DS1, "grid": "o96"}]
    assert p.output_statistics == "y.npz"


def test_parse_old_flags_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown option"):
        _parse([DS0, "--tendencies", "6h"])
    with pytest.raises(ValueError, match="Unknown option"):
        _parse([DS0, "--residual", DS1])


def test_default_action_is_statistics() -> None:
    assert _parse([DS0]).do_statistics is True


def test_parse_global_start_end_frequency() -> None:
    # The flags are applied to the dataset and to the one given to --minus, so
    # that the two do not have to repeat them.
    p = _parse([DS0, "--minus", DS1, "--start", "2021-03-01", "--end", "2021-03-31", "--frequency", "12h"])
    for kwargs in (p.open_kwargs, p.residual_open_kwargs):
        assert kwargs == {"start": "2021-03-01", "end": "2021-03-31", "frequency": "12h"}


def test_parse_global_dates_reach_a_json_config() -> None:
    cfg = json.dumps({"dataset": DS0, "select": ["a"]})
    p = _parse([cfg, "--start", "2021"])
    # A JSON config takes no key=value, but open_dataset merges the kwargs.
    assert p.open_args == [{"dataset": DS0, "select": ["a"]}]
    assert p.open_kwargs == {"start": 2021}


@pytest.mark.parametrize(
    "tokens",
    [
        [DS0, "start=2021-01-01", "--start", "2021-03-01"],
        [json.dumps({"dataset": DS0, "end": "2021-12-31"}), "--end", "2021-03-31"],
        [DS0, "--minus", DS1, "frequency=6h", "--frequency", "12h"],
    ],
)
def test_parse_global_dates_conflict(tokens) -> None:
    with pytest.raises(ValueError, match="conflicts with the"):
        _parse(tokens)


def test_parse_output_flags() -> None:
    p = _parse([DS0, "--output", "gen.zarr", "--output-statistics", "stats.npz"])
    assert p.output_dataset == "gen.zarr"
    assert p.output_statistics == "stats.npz"

    # The statistics file is a numpy archive, and only that.
    with pytest.raises(ValueError, match="must end with '.npz'"):
        _parse([DS0, "--output-statistics", "stats.json"])

    # --output-dataset is the old name of --output, still accepted.
    assert _parse([DS0, "--output-dataset", "gen.zarr"]).output_dataset == "gen.zarr"


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
def test_global_dates_equal_the_per_dataset_options(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def statistics(name: str, rest: list[str]) -> dict:
        out = tmp_path / f"{name}.npz"
        Compute().run(argparse.Namespace(rest=rest + ["--output-statistics", str(out)]))
        return load_document(str(out))["statistics"]

    window = ["start=2021-03-01", "end=2021-03-31"]
    per_dataset = statistics("per-dataset", [DS0] + window + ["--minus", DS1] + window)
    globally = statistics("global", [DS0, "--minus", DS1, "--start", "2021-03-01", "--end", "2021-03-31"])

    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(per_dataset[key], globally[key])

    # ... and they really did restrict the period.
    everything = statistics("everything", [DS0])
    assert not np.allclose(everything["mean"], globally["mean"])


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
        col.update(_read_block(ds, None, blocks[b][0]))
    ckpt = "/tmp/compute_test_resume.pkl"
    _save_checkpoint(
        ckpt,
        {
            "version": CHECKPOINT_VERSION,
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
# Missing dates
# --------------------------------------------------------------------------- #

# Dates of DS0 (6-hourly from 2021-01-01T00) forced missing, and their indices.
MISSING_DATES = ["2021-01-01T06:00:00", "2021-01-02T00:00:00", "2021-01-02T06:00:00"]
MISSING_INDICES = frozenset({1, 4, 5})


def _missing_task(**kwargs) -> Task:
    """A task on DS0 with a few of its dates forced missing."""
    base = dict(open_kwargs={"set_missing_dates": MISSING_DATES}, args_sha="missing")
    base.update(kwargs)
    return _make_task(**base)


def _expected_tendency(data: np.ndarray, delta: int, missing: frozenset) -> np.ndarray:
    """Tendencies over the runs of consecutive readable dates, as numpy sees them."""
    rows = [data[i] - data[i - delta] for lo, hi in _runs(0, len(data), missing) for i in range(lo + delta, hi)]
    return np.stack(rows)


def _flat_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    flat = np.moveaxis(values, 1, 0).reshape(values.shape[1], -1)
    return {
        "mean": np.nanmean(flat, axis=1),
        "stdev": np.nanstd(flat, axis=1),
        "minimum": np.nanmin(flat, axis=1),
        "maximum": np.nanmax(flat, axis=1),
    }


def test_runs_splits_on_missing_dates() -> None:
    assert _runs(0, 10, frozenset()) == [(0, 10)]
    assert _runs(0, 0, frozenset()) == []
    assert _runs(0, 10, frozenset({0})) == [(1, 10)]
    assert _runs(0, 10, frozenset({9})) == [(0, 9)]
    assert _runs(0, 10, frozenset({3, 4, 7})) == [(0, 3), (5, 7), (8, 10)]
    assert _runs(2, 6, frozenset({0, 4, 9})) == [(2, 4), (5, 6)]
    assert _runs(0, 3, frozenset({0, 1, 2})) == []


def test_blocks_skip_missing_dates_and_flag_the_runs() -> None:
    # Runs are (0, 3), (5, 7), (8, 10); a chunk of 2 splits the first one.
    assert _blocks(10, 2, None, frozenset({3, 4, 7})) == [
        (slice(0, 2), True),
        (slice(2, 3), False),
        (slice(5, 7), True),
        (slice(8, 10), True),
    ]
    # Subsampled blocks simply drop the missing indices.
    assert _blocks(10, 3, np.array([1, 3, 5, 7, 9]), frozenset({3, 7})) == [
        ([1, 5, 9], True),
    ]


@mockup_open_zarr
def test_missing_dates_are_skipped() -> None:
    _, results = run_engine(_missing_task(do_statistics=True, chunk_size=7))

    data = np.asarray(open_dataset(DS0)[:], dtype=np.float64)
    kept = np.delete(data, sorted(MISSING_INDICES), axis=0)
    expected = _flat_stats(kept)
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(results["statistics"][key], expected[key])


@pytest.mark.parametrize("delta,steps", [("6h", 1), ("12h", 2)])
@mockup_open_zarr
def test_missing_dates_tendencies_do_not_span_the_gap(delta: str, steps: int) -> None:
    _, results = run_engine(_missing_task(do_statistics=False, tendency=delta, chunk_size=7))

    data = np.asarray(open_dataset(DS0)[:], dtype=np.float64)
    expected = _flat_stats(_expected_tendency(data, steps, MISSING_INDICES))
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(results["tendency"][key], expected[key])


@mockup_open_zarr
def test_missing_dates_parallel_equals_sequential() -> None:
    _, seq = run_engine(_missing_task(do_statistics=True, tendency="12h", chunk_size=7))
    _, par = run_engine(_missing_task(do_statistics=True, tendency="12h", chunk_size=7, parallel=4))
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(seq["statistics"][key], par["statistics"][key])
        np.testing.assert_allclose(seq["tendency"][key], par["tendency"][key])


@mockup_open_zarr
def test_missing_date_next_to_a_segment_boundary() -> None:
    # With 4 workers the 1460 dates are cut into 16 segments of 92, so index 90
    # sits two rows before a segment boundary: the worker of the second segment
    # must seed its tendency window with row 91 only, as the sequential loop does.
    kwargs = {"open_kwargs": {"set_missing_dates": ["2021-01-23T12:00:00"]}, "args_sha": "boundary"}
    _, seq = run_engine(_make_task(do_statistics=True, tendency="12h", chunk_size=1, **kwargs))
    _, par = run_engine(_make_task(do_statistics=True, tendency="12h", chunk_size=1, parallel=4, **kwargs))

    data = np.asarray(open_dataset(DS0)[:], dtype=np.float64)
    expected = _flat_stats(_expected_tendency(data, 2, frozenset({90})))
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(seq["tendency"][key], expected[key])
        np.testing.assert_allclose(par["tendency"][key], expected[key])


@mockup_open_zarr
def test_missing_dates_of_either_dataset_are_skipped() -> None:
    # Only the subtracted dataset has missing dates; they are still skipped, and
    # the residual stays the constant -0.1 (DS0 and DS1 differ by k only).
    task = _make_task(
        do_statistics=True,
        has_residual=True,
        residual_open_args=[DS1],
        residual_open_kwargs={"set_missing_dates": MISSING_DATES},
        residual_label=DS1,
    )
    _, results = run_engine(task)
    n = len(open_dataset(DS0).variables)
    np.testing.assert_allclose(results["statistics"]["mean"], np.full(n, -0.1), atol=1e-5)
    np.testing.assert_allclose(results["statistics"]["stdev"], np.zeros(n), atol=1e-5)


@mockup_open_zarr
def test_all_dates_missing_is_an_error() -> None:
    dates = [str(d) for d in open_dataset(DS0).dates]
    task = _make_task(do_statistics=True, open_kwargs={"set_missing_dates": dates})
    with pytest.raises(ValueError, match="are missing; there is nothing to compute"):
        run_engine(task)


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

    # First refresh: no previous snapshot, so no deltas.
    prev = _render_live(col, list(ds.variables), [0, 1, 2], None)
    out = capsys.readouterr().out
    assert "Statistics" in out
    assert ds.variables[0] in out
    assert prev is not None

    # Second refresh: deltas are shown as signed values in parentheses.
    col.update(np.asarray(ds[4:8], dtype=np.float64))
    _render_live(col, list(ds.variables), [0, 1, 2], prev)
    out = capsys.readouterr().out
    assert "(" in out and (")" in out)


@mockup_open_zarr
def test_run_with_live_enabled_smoke() -> None:
    _, results = run_engine(_make_task(do_statistics=True, live=True, chunk_size=200))
    assert results["statistics"] is not None


# --------------------------------------------------------------------------- #
# Full command (CLI entry) with JSON output
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_command_writes_the_statistics_archive(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # keep the default checkpoint out of the repo
    out = tmp_path / "out.npz"
    Compute().run(argparse.Namespace(rest=[DS0, "--statistics", "--output-statistics", str(out)]))
    doc = load_document(str(out))
    assert doc["dataset"] == DS0
    assert doc["statistics"] is not None
    assert len(doc["statistics"]["mean"]) == len(open_dataset(DS0).variables)


def test_default_output_name() -> None:
    assert _default_statistics_output(_parse(["/data/foo.zarr", "--statistics"])) == "foo.statistics.npz"
    assert _default_statistics_output(_parse(["bar", "--statistics"])) == "bar.statistics.npz"


@mockup_open_zarr
def test_default_output_and_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    default = tmp_path / f"{DS0}.statistics.npz"

    Compute().run(argparse.Namespace(rest=[DS0, "--statistics"]))
    assert default.exists()
    assert load_document(str(default))["dataset"] == DS0

    # Re-running without --overwrite must fail before recomputing.
    with pytest.raises(ValueError, match="already exists"):
        Compute().run(argparse.Namespace(rest=[DS0, "--statistics"]))

    # --overwrite replaces it.
    Compute().run(argparse.Namespace(rest=[DS0, "--statistics", "--overwrite"]))
    assert default.exists()


@mockup_open_zarr
def test_command_json_equals_name_form(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    name_out = tmp_path / "name.npz"
    json_out = tmp_path / "json.npz"
    Compute().run(argparse.Namespace(rest=[DS0, "--statistics", "--output-statistics", str(name_out)]))
    cfg = json.dumps({"dataset": DS0})
    Compute().run(argparse.Namespace(rest=[cfg, "--statistics", "--output-statistics", str(json_out)]))
    a = load_document(str(name_out))["statistics"]["mean"]
    b = load_document(str(json_out))["statistics"]["mean"]
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


# --------------------------------------------------------------------------- #
# Generated dataset (--output)
# --------------------------------------------------------------------------- #


def test_parse_output_dataset() -> None:
    p = _parse([DS0, "--output", "out.zarr"])
    assert p.output_dataset == "out.zarr"
    # A generated dataset always carries its statistics.
    p = _parse([DS0, "--statistics-tendencies", "6h", "--output", "out.zarr"])
    assert p.do_statistics is True


def test_parse_output_dataset_rejects_bad_combinations() -> None:
    with pytest.raises(ValueError, match="must end with"):
        _parse([DS0, "--output", "out"])
    with pytest.raises(ValueError, match="cannot be combined with --sample-dates"):
        _parse([DS0, "--output", "out.zarr", "--sample-dates", "0.1"])


def test_check_dataset_rejects_unsupported_views() -> None:
    class _Fake:
        def __init__(self, shape, missing):
            self.shape = shape
            self.missing = missing

    OutputDataset.check_dataset(_Fake((10, 4, 1, 7), set()), "ok")

    with pytest.raises(ValueError, match="expected a gridded dataset"):
        OutputDataset.check_dataset(_Fake((10, 4), set()), "flat")

    with pytest.raises(ValueError, match="no dates"):
        OutputDataset.check_dataset(_Fake((0, 4, 1, 7), set()), "empty")

    # Missing dates are accepted: they are written as NaN and recorded as the
    # generated dataset's own missing dates.
    OutputDataset.check_dataset(_Fake((10, 4, 1, 7), {3}), "holes")


def _generate(rest: list[str], tmp_path) -> tuple[Any, dict]:
    """Run the command inside the mockup, then open the generated dataset outside it.

    The mockup patches ``zarr.open`` and the dataset lookup, so the generated store
    can only be read once it is no longer active.
    """

    @mockup_open_zarr
    def build() -> dict:
        Compute().run(argparse.Namespace(rest=rest))
        source = open_dataset(DS0)
        return {
            "data": np.asarray(source[:]),
            "dates": source.dates,
            "latitudes": np.asarray(source.latitudes),
            "longitudes": np.asarray(source.longitudes),
            "variables": list(source.variables),
            "resolution": source.resolution,
            "field_shape": tuple(source.field_shape),
            "frequency": source.frequency,
        }

    return build()


def test_generated_dataset_matches_the_source(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "gen.zarr"
    doc_path = tmp_path / "gen.npz"

    source = _generate(
        [DS0, "--statistics", "--statistics-tendencies", "6h", "--chunk-size", "7"]
        + ["--output", str(out), "--output-statistics", str(doc_path)],
        tmp_path,
    )

    ds = open_dataset(str(out))

    # The values, the time axis and the grid are those of the dataset as opened.
    np.testing.assert_allclose(np.asarray(ds[:]), source["data"].astype("float32"))
    np.testing.assert_array_equal(ds.dates, source["dates"])
    np.testing.assert_allclose(ds.latitudes, source["latitudes"])
    np.testing.assert_allclose(ds.longitudes, source["longitudes"])
    assert list(ds.variables) == source["variables"]
    assert ds.resolution == source["resolution"]
    assert tuple(ds.field_shape) == source["field_shape"]
    assert ds.frequency == source["frequency"]

    # The recomputed statistics are the dataset's statistics.
    doc = load_document(str(doc_path))
    assert doc["output_dataset"] == str(out)
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(ds.statistics[key], doc["statistics"][key])
        np.testing.assert_allclose(ds.statistics_tendencies("6h")[key], doc["tendency_statistics"][key])

    # The metadata is complete enough to be used without a recipe.
    attrs = dict(zarr.open(str(out), mode="r").attrs)
    assert attrs["layout"] == "gridded"
    assert attrs["missing_dates"] == []
    assert attrs["dtype"] == "float32"
    assert attrs["constant_fields"] == []
    assert attrs["uuid"]
    assert attrs["version"]
    assert attrs["start_date"] == source["dates"][0].astype(object).isoformat()
    assert attrs["end_date"] == source["dates"][-1].astype(object).isoformat()
    assert attrs["derived_from"]["arithmetic"] == "datasets[0]"
    assert [d["label"] for d in attrs["derived_from"]["datasets"]] == [DS0]

    # The whole metadata chain works on the generated dataset.
    assert ds.metadata()["variables"] == source["variables"]


def _make_trajectories(path, **kwargs) -> None:
    """Write a small trajectories dataset to ``path``."""
    from test_trajectories import make_trajectories_zarr

    make_trajectories_zarr(path=str(path), analytic=True, **kwargs)


def test_generated_trajectories_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "traj-src.zarr"
    out = tmp_path / "traj-gen.zarr"
    doc_path = tmp_path / "traj.npz"
    _make_trajectories(src, n_dates=4, n_steps=5, n_vars=3, n_cells=10)

    source = open_dataset(str(src))
    expected = np.asarray(source[:])

    Compute().run(
        argparse.Namespace(
            rest=[
                str(src),
                "--statistics",
                "--chunk-size",
                "2",
                "--output",
                str(out),
                "--output-statistics",
                str(doc_path),
            ]
        )
    )

    ds = open_dataset(str(out))

    # The generated store has the trajectories layout, with both time axes.
    assert zarr.open(str(out), mode="r").attrs["layout"] == "trajectories"
    assert ds.shape == source.shape
    np.testing.assert_array_equal(ds.base_dates, source.base_dates)
    np.testing.assert_array_equal(ds.steps, source.steps)
    assert ds.base_frequency == source.base_frequency
    assert ds.step_frequency == source.step_frequency
    assert list(ds.variables) == list(source.variables)
    np.testing.assert_allclose(ds.latitudes, source.latitudes)
    np.testing.assert_allclose(ds.longitudes, source.longitudes)

    # One base date, one ensemble and one step per chunk, so no two workers of a
    # parallel run ever write to the same chunk.
    raw = zarr.open(str(out), mode="r")["data"]
    assert raw.chunks == (1, len(source.variables), 1, 1, 10)

    # The values are those of the source, and so are the statistics.
    np.testing.assert_allclose(np.asarray(ds[:]), expected.astype("float32"))
    doc = load_document(str(doc_path))
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(ds.statistics[key], doc["statistics"][key])

    # The envelope is on valid times: base date + step.
    attrs = zarr.open(str(out), mode="r").attrs
    assert attrs["start_date"] == str(source.base_dates[0] + source.steps[0])
    assert attrs["end_date"] == str(source.base_dates[-1] + source.steps[-1])
    assert attrs["start_base_date"] == str(source.base_dates[0])
    assert attrs["end_base_date"] == str(source.base_dates[-1])
    assert attrs["dimensions"] == ["base_dates", "variables", "ensembles", "steps", "values"]

    # ... and the metadata of the generated dataset serialises end to end.
    md = ds.dataset_metadata()
    assert md["base_frequency"] == "6h"
    assert md["step_frequency"] == "6h"
    ds.metadata()


def test_generated_trajectories_dataset_parallel_equals_sequential(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "traj-par-src.zarr"
    _make_trajectories(src, n_dates=6, n_steps=4, n_vars=3, n_cells=10)

    outputs = []
    for name, extra in (("seq", []), ("par", ["--parallel", "3"])):
        out = tmp_path / f"traj-{name}.zarr"
        Compute().run(
            argparse.Namespace(
                rest=[str(src), "--output", str(out), "--output-statistics", str(tmp_path / f"{name}.npz")] + extra
            )
        )
        outputs.append(open_dataset(str(out)))

    seq, par = outputs
    np.testing.assert_array_equal(np.asarray(seq[:]), np.asarray(par[:]))
    np.testing.assert_array_equal(np.asarray(seq[:]), np.asarray(open_dataset(str(src))[:]).astype("float32"))
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(seq.statistics[key], par.statistics[key])


def test_generated_trajectories_residual_with_missing_dates(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    a = tmp_path / "traj-a.zarr"
    b = tmp_path / "traj-b.zarr"
    out = tmp_path / "traj-res.zarr"
    doc_path = tmp_path / "traj-res.npz"

    # 'b' is missing its second base date; 'a' is complete.
    _make_trajectories(a, n_dates=4, n_steps=3, n_vars=2, n_cells=6)
    _make_trajectories(b, n_dates=4, n_steps=3, n_vars=2, n_cells=6, missing_dates=["2021-01-01T06:00:00"])

    Compute().run(
        argparse.Namespace(rest=[str(a), "--minus", str(b), "--output", str(out), "--output-statistics", str(doc_path)])
    )

    ds = open_dataset(str(out))

    # The gap of the subtracted dataset is the gap of the residual.
    assert ds.missing == {1}

    from anemoi.datasets import MissingDateError

    with pytest.raises(MissingDateError):
        ds[1]

    # Both are the same data, so the residual is exactly zero where it is defined.
    for i in (0, 2, 3):
        np.testing.assert_allclose(np.asarray(ds[i]), 0.0)

    raw = zarr.open(str(out), mode="r")["data"]
    assert np.isnan(np.asarray(raw[1])).all()

    doc = load_document(str(doc_path))
    np.testing.assert_allclose(doc["statistics"]["mean"], 0.0)


def test_trajectories_reject_tendencies(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "traj-tend.zarr"
    _make_trajectories(src, n_dates=3, n_steps=2, n_vars=2, n_cells=4)

    with pytest.raises(ValueError, match="two time axes"):
        Compute().run(
            argparse.Namespace(
                rest=[str(src), "--statistics-tendencies", "6h", "--output-statistics", str(tmp_path / "t.npz")]
            )
        )


def test_generated_dataset_keeps_the_missing_dates(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "gen-missing.zarr"
    doc_path = tmp_path / "gen-missing.npz"
    config = json.dumps({"dataset": DS0, "set_missing_dates": MISSING_DATES})

    source = _generate(
        [config, "--statistics", "--chunk-size", "7", "--output", str(out), "--output-statistics", str(doc_path)],
        tmp_path,
    )

    ds = open_dataset(str(out))

    # The generated dataset has the gaps of the one it was generated from.
    assert ds.missing == set(MISSING_INDICES)
    assert len(ds) == len(source["dates"])
    np.testing.assert_array_equal(ds.dates, source["dates"])

    from anemoi.datasets import MissingDateError

    with pytest.raises(MissingDateError):
        ds[sorted(MISSING_INDICES)[0]]

    # The readable dates hold the values, the missing ones are NaN on disk.
    present = [i for i in range(len(ds)) if i not in MISSING_INDICES]
    for i in (0, 2, 3, len(ds) - 1):
        np.testing.assert_allclose(np.asarray(ds[i]), source["data"][i].astype("float32"))
    np.testing.assert_allclose(np.asarray(ds[6:20]), source["data"][6:20].astype("float32"))

    raw = zarr.open(str(out), mode="r")["data"]
    assert np.isnan(np.asarray(raw[sorted(MISSING_INDICES)])).all()

    # The statistics are those of the readable dates only.
    doc = load_document(str(doc_path))
    expected = _flat_stats(source["data"][present].astype(np.float64))
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(ds.statistics[key], doc["statistics"][key])
        np.testing.assert_allclose(ds.statistics[key], expected[key], rtol=1e-6)


def test_generated_dataset_reflects_the_view(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "view.zarr"

    _generate(
        [DS0, "select=a", "start=2021-03-01", "end=2021-03-31", "--output", str(out)],
        tmp_path,
    )

    ds = open_dataset(str(out))
    assert list(ds.variables) == ["a"]
    assert ds.shape[1] == 1
    assert str(ds.dates[0]).startswith("2021-03-01")
    assert str(ds.dates[-1]).startswith("2021-03-31")


def test_generated_residual_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "residual.zarr"

    @mockup_open_zarr
    def build() -> np.ndarray:
        Compute().run(
            argparse.Namespace(
                rest=[DS0, "--minus", DS1, "--chunk-size", "5", "--parallel", "3"] + ["--output", str(out)]
            )
        )
        a = np.asarray(open_dataset(DS0)[:], dtype=np.float64)
        b = np.asarray(open_dataset(DS1)[:], dtype=np.float64)
        return a - b

    difference = build()

    ds = open_dataset(str(out))
    # DS0 and DS1 differ by 0.1 everywhere; float32 storage of values around 2e9
    # is why this is only approximate.
    np.testing.assert_allclose(np.asarray(ds[:]), difference.astype("float32"), atol=1e-3)
    np.testing.assert_allclose(ds.statistics["mean"], np.full(len(ds.variables), -0.1), atol=1e-5)

    attrs = dict(zarr.open(str(out), mode="r").attrs)
    assert attrs["derived_from"]["arithmetic"] == "datasets[0] - datasets[1]"
    assert [d["label"] for d in attrs["derived_from"]["datasets"]] == [DS0, DS1]
    assert "Residual" in attrs["description"]


def test_generated_dataset_parallel_equals_sequential(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sequential = tmp_path / "seq.zarr"
    parallel = tmp_path / "par.zarr"

    @mockup_open_zarr
    def build() -> None:
        for path, extra in ((sequential, []), (parallel, ["--parallel", "3"])):
            Compute().run(
                argparse.Namespace(
                    rest=[DS0, "--statistics", "--chunk-size", "5"]
                    + ["--output", str(path), "--output-statistics", f"{path}.npz"]
                    + extra
                )
            )

    build()

    a, b = open_dataset(str(sequential)), open_dataset(str(parallel))
    np.testing.assert_array_equal(np.asarray(a[:]), np.asarray(b[:]))
    np.testing.assert_allclose(a.statistics["mean"], b.statistics["mean"])
    assert a.constant_fields == b.constant_fields


def test_generated_dataset_refuses_to_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "once.zarr"

    @mockup_open_zarr
    def build(document: str, extra: list[str]) -> None:
        Compute().run(
            argparse.Namespace(
                rest=[DS0, "--statistics", "--output", str(out), "--output-statistics", document] + extra
            )
        )

    build(str(tmp_path / "first.npz"), [])

    # The store is there, and is only replaced when --overwrite is given.
    with pytest.raises(ValueError, match="Output dataset already exists"):
        build(str(tmp_path / "second.npz"), [])

    build(str(tmp_path / "third.npz"), ["--overwrite"])
    assert open_dataset(str(out)).shape[0] > 0


@pytest.mark.parametrize("document", ["same.npz"])
def test_overwrite_replaces_both_outputs(tmp_path, monkeypatch, document) -> None:
    # Re-running the very same command: --overwrite must replace the results file
    # *and* the generated dataset, in both output formats.
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "same.zarr"
    doc = tmp_path / document

    @mockup_open_zarr
    def build(extra: list[str]) -> None:
        Compute().run(
            argparse.Namespace(
                rest=[DS0, "--statistics", "--output", str(out), "--output-statistics", str(doc)] + extra
            )
        )

    build([])
    assert doc.exists()
    first = doc.stat().st_mtime_ns

    with pytest.raises(ValueError, match="Statistics file already exists"):
        build([])

    # The refusal happens before anything is computed, so nothing was touched.
    assert doc.stat().st_mtime_ns == first

    build(["--overwrite"])

    ds = open_dataset(str(out))
    stats = load_document(str(doc))["statistics"]
    for key in ("mean", "stdev", "minimum", "maximum"):
        np.testing.assert_allclose(ds.statistics[key], stats[key])


# --------------------------------------------------------------------------- #
# Constants tracking (no dataset)
# --------------------------------------------------------------------------- #


def test_constants_tracker() -> None:
    variables = ["constant", "varying", "nans"]
    rows = np.zeros((6, 3, 1, 4))
    rows[:, 0] = 3.0  # constant in time
    rows[:, 1] = np.arange(6).reshape(6, 1, 1)  # varies in time
    rows[:, 2] = np.nan  # constant, all NaNs

    tracker = ConstantsTracker(variables)
    tracker.update(rows)
    assert tracker.constant_fields() == ["constant", "nans"]

    # A tracker that has seen nothing reports nothing.
    assert ConstantsTracker(variables).constant_fields() == []


def test_constants_tracker_merge_equivalence() -> None:
    variables = ["a", "b"]
    rows = np.zeros((8, 2, 1, 3))
    rows[:, 0] = 1.5
    rows[:, 1] = np.arange(8).reshape(8, 1, 1)

    whole = ConstantsTracker(variables)
    whole.update(rows)

    first, second = ConstantsTracker(variables), ConstantsTracker(variables)
    first.update(rows[:4])
    second.update(rows[4:])
    assert first.merge(second).constant_fields() == whole.constant_fields()

    # Two segments that are each constant but differ from each other are not
    # constant once merged.
    left, right = ConstantsTracker(["x"]), ConstantsTracker(["x"])
    left.update(np.full((3, 1, 1, 2), 1.0))
    right.update(np.full((3, 1, 1, 2), 2.0))
    assert left.constant_fields() == ["x"]
    assert right.constant_fields() == ["x"]
    assert left.merge(right).constant_fields() == []


def test_parse_output_dataset_dtype_is_not_an_option() -> None:
    # The generated dataset is always float32; there is no way to ask for another.
    with pytest.raises(ValueError, match="Unknown option"):
        _parse([DS0, "--output", "out.zarr", "--dataset-dtype", "float64"])


# --------------------------------------------------------------------------- #
# Interpolation to a grid (--grid)
# --------------------------------------------------------------------------- #


def test_parse_grid() -> None:
    p = _parse([DS0, "--grid", "o96"])
    assert p.grid == "o96"
    assert p.grid_method == "nearest"

    p = _parse([DS0, "--minus", DS1, "--grid", "0.25", "--grid-method", "linear"])
    assert p.grid == "0.25"
    assert p.grid_method == "linear"
    assert p.residual_open_args == [DS1]

    with pytest.raises(ValueError, match="--grid-method requires --grid"):
        _parse([DS0, "--grid-method", "linear"])


def test_lookup_key() -> None:
    assert _lookup_key("o96") == "o96"
    assert _lookup_key("n320") == "n320"
    assert _lookup_key("/tmp/grid-o96.npz") == "/tmp/grid-o96.npz"
    assert _lookup_key("0.25") == [0.25, 0.25]
    assert _lookup_key("0.25x0.5") == [0.25, 0.5]
    assert _lookup_key("1/2") == [1.0, 2.0]


def _grid_file(path, latitudes, longitudes) -> str:
    """Write an ``.npz`` grid file, so that no grid has to be downloaded."""
    np.savez(path, latitudes=np.asarray(latitudes), longitudes=np.asarray(longitudes))
    return str(path)


def test_target_grid_name() -> None:
    assert TargetGrid.name.fget(argparse.Namespace(spec="o96")) == "o96"
    assert TargetGrid.name.fget(argparse.Namespace(spec="/data/grid-n320.npz")) == "n320"
    assert TargetGrid.name.fget(argparse.Namespace(spec="/data/mygrid.npz")) == "mygrid"


@mockup_open_zarr
def test_interpolator_is_skipped_on_the_same_grid(tmp_path) -> None:
    ds = open_dataset(DS0)
    same = TargetGrid(_grid_file(tmp_path / "same.npz", ds.latitudes, ds.longitudes))
    assert Interpolator.build(ds, same) is None
    assert Interpolator.build(ds, None) is None


@mockup_open_zarr
def test_interpolator_nearest_picks_the_source_points(tmp_path) -> None:
    ds = open_dataset(DS0)
    latitudes, longitudes = np.asarray(ds.latitudes), np.asarray(ds.longitudes)

    # A target made of every other source point: nearest neighbours are exact.
    target = TargetGrid(_grid_file(tmp_path / "half.npz", latitudes[::2], longitudes[::2]))
    interpolator = Interpolator.build(ds, target)
    assert interpolator is not None

    data = np.asarray(ds[0:4], dtype=np.float64)
    np.testing.assert_array_equal(interpolator(data), data[..., ::2])

    with pytest.raises(ValueError, match="points on its grid axis"):
        interpolator(data[..., :3])


def test_generated_dataset_on_another_grid(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "regridded.zarr"

    @mockup_open_zarr
    def build() -> dict:
        source = open_dataset(DS0)
        grid = _grid_file(tmp_path / "half.npz", np.asarray(source.latitudes)[::2], np.asarray(source.longitudes)[::2])
        Compute().run(
            argparse.Namespace(
                rest=[DS0, "--statistics", "--grid", grid, "--output", str(out), "--output-statistics", "g.npz"]
            )
        )
        return {"data": np.asarray(source[:]), "latitudes": np.asarray(source.latitudes)}

    source = build()

    ds = open_dataset(str(out))
    # Nearest neighbours onto a subset of the source points reproduce them exactly.
    np.testing.assert_allclose(np.asarray(ds[:]), source["data"][..., ::2].astype("float32"))
    np.testing.assert_allclose(ds.latitudes, source["latitudes"][::2])
    assert ds.shape[-1] == len(source["latitudes"][::2])

    attrs = dict(zarr.open(str(out), mode="r").attrs)
    assert attrs["resolution"] == "half"
    assert attrs["field_shape"] == [len(source["latitudes"][::2])]
    assert attrs["derived_from"]["computation"]["grid_method"] == "nearest"
    # Attributes describing the source grid are not carried over.
    assert "data_request" not in attrs
    assert "proj_string" not in attrs


@mockup_open_zarr
def test_minus_needs_a_common_grid(tmp_path) -> None:
    # DS0 has 10 grid points, DS_SMALL has 5.
    task = _make_task(
        do_statistics=True,
        has_residual=True,
        residual_open_args=[DS_SMALL],
        residual_open_kwargs={},
        residual_label=DS_SMALL,
    )
    with pytest.raises(ValueError, match="different field shapes"):
        run_engine(task)


def test_minus_on_a_common_grid(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "difference.zarr"

    @mockup_open_zarr
    def build() -> dict:
        a, b = open_dataset(DS0), open_dataset(DS_SMALL)
        # Interpolate both onto the grid of the smaller dataset: B is left alone.
        grid = _grid_file(tmp_path / "small.npz", b.latitudes, b.longitudes)
        Compute().run(
            argparse.Namespace(
                rest=[DS0, "--minus", DS_SMALL, "--grid", grid]
                + ["--output", str(out), "--output-statistics", "d.npz", "--chunk-size", "5"]
            )
        )
        target = TargetGrid(grid)
        interpolator = Interpolator.build(a, target)
        assert Interpolator.build(b, target) is None, "B is already on the target grid"
        return {
            "difference": interpolator(np.asarray(a[:], dtype=np.float64)) - np.asarray(b[:], dtype=np.float64),
            "latitudes": np.asarray(b.latitudes),
        }

    expected = build()

    ds = open_dataset(str(out))
    np.testing.assert_allclose(np.asarray(ds[:]), expected["difference"].astype("float32"), rtol=1e-6)
    np.testing.assert_allclose(ds.latitudes, expected["latitudes"])

    attrs = dict(zarr.open(str(out), mode="r").attrs)
    assert attrs["derived_from"]["arithmetic"] == "datasets[0] - datasets[1]"
    assert "interpolated to" in attrs["description"]
