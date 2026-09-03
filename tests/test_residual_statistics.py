# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``open_dataset(..., residual_statistics=...)`` and ``ds.residual_statistics``.

The residual-statistics JSON file is produced by ``anemoi-datasets compute
<a> --minus <b>``; these tests cover the round trip (writer to
reader), the validation of the file, and the propagation of
``residual_statistics`` through the wrapping datasets.
"""

import argparse
import datetime
import json

import numpy as np
import pytest
from test_data import mockup_open_zarr

from anemoi.datasets import open_dataset
from anemoi.datasets.commands.compute import Compute
from anemoi.datasets.misc.residual_statistics import METADATA_KEY
from anemoi.datasets.misc.residual_statistics import RESIDUAL_KIND
from anemoi.datasets.misc.residual_statistics import STATISTICS_KEYS
from anemoi.datasets.misc.residual_statistics import VERSION
from anemoi.datasets.misc.residual_statistics import ResidualStatisticsFile
from anemoi.datasets.misc.residual_statistics import ResidualStatisticsNotAvailable
from anemoi.datasets.misc.residual_statistics import load_document
from anemoi.datasets.misc.residual_statistics import save

# Two fake datasets differing only by their `k` index, so that the residual
# DS0 - DS1 is the same constant for every variable and every point.
DS0 = "test-2021-2021-6h-o96-abcd-0"
DS1 = "test-2021-2021-6h-o96-abcd-1"

VARIABLES = ["a", "b", "c", "d"]


def write_file(path, *, variables=VARIABLES, kind=RESIDUAL_KIND, datasets=(DS0, DS1), values=None, **overrides):
    """Write a residual-statistics archive by hand and return its path.

    Written at the numpy level rather than through ``save()`` so that invalid
    files -- a missing ``kind``, the wrong number of values, ... -- can be
    written too.
    """
    if values is None:
        values = {k: [float(i) for i in range(len(variables))] for k in STATISTICS_KEYS}

    document = {"version": VERSION, "datasets": list(datasets)}
    if kind is not None:
        document["kind"] = kind
    document["variables"] = list(variables)
    document.update(overrides)

    arrays = {}
    if isinstance(values, dict):
        for key, v in values.items():
            # A statistic that is not a list of numbers cannot be an array, so it
            # goes into the metadata instead, where the reader still finds it.
            if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                arrays[f"statistics.{key}"] = np.asarray(v, dtype=np.float64)
            else:
                document.setdefault("statistics", {})[key] = v
    else:
        document["statistics"] = values

    np.savez(str(path), metadata=np.array(json.dumps(document)), **arrays)
    return str(path)


def compute_residual(tmp_path, *extra, name="residual.npz"):
    """Run the ``compute`` command to produce a real residual file."""
    out = tmp_path / name
    Compute().run(argparse.Namespace(rest=[DS0, "--minus", DS1, "--output-statistics", str(out), *extra]))
    return str(out)


# --------------------------------------------------------------------------- #
# Round trip: the compute command writes what open_dataset can read
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_compute_writes_a_readable_residual_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # keep the default checkpoint out of the repo
    path = compute_residual(tmp_path)

    document = load_document(path)
    assert document["kind"] == RESIDUAL_KIND
    assert document["version"] == VERSION
    assert document["datasets"] == [DS0, DS1]

    ds = open_dataset(DS0, residual_statistics=path)
    assert sorted(ds.residual_statistics) == sorted(STATISTICS_KEYS)
    for key in STATISTICS_KEYS:
        assert len(ds.residual_statistics[key]) == len(ds.variables)

    # DS0 - DS1 is the constant -0.1 everywhere, so the residual has no spread.
    np.testing.assert_allclose(ds.residual_statistics["mean"], -0.1, atol=1e-6)
    np.testing.assert_allclose(ds.residual_statistics["stdev"], 0.0, atol=1e-6)


@mockup_open_zarr
def test_plain_statistics_file_is_marked_as_such(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "plain.npz"
    Compute().run(argparse.Namespace(rest=[DS0, "--statistics", "--output-statistics", str(out)]))
    document = load_document(str(out))
    assert document["kind"] == "statistics"
    assert document["datasets"] == [DS0]

    with pytest.raises(ValueError, match="plain statistics, not residual"):
        open_dataset(DS0, residual_statistics=str(out))


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_document_records_its_provenance(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = compute_residual(tmp_path, "--statistics")

    import anemoi.datasets

    provenance = load_document(path)["derived_from"]
    assert provenance["command"] == "anemoi-datasets compute"
    assert provenance["anemoi_datasets_version"] == anemoi.datasets.__version__
    assert provenance["arithmetic"] == "datasets[0] - datasets[1]"
    assert [d["label"] for d in provenance["datasets"]] == [DS0, DS1]
    assert provenance["datasets"][0]["open_args"] == [DS0]
    assert provenance["computation"]["grid"] is None
    # sys.argv of the process that wrote it (pytest's, here, not the CLI's).
    assert provenance["computation"]["command_line"]
    datetime.datetime.fromisoformat(provenance["created"])  # a real timestamp


@mockup_open_zarr
def test_provenance_reaches_the_dataset_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = compute_residual(tmp_path)
    ds = open_dataset(DS0, residual_statistics=path)

    import anemoi.datasets

    recorded = ds.metadata_specific()["residual_statistics"]
    # Only the two scalars, not the whole entry: this lands in checkpoints.
    assert recorded["anemoi_datasets_version"] == anemoi.datasets.__version__
    assert recorded["created"]
    assert "datasets" in recorded and "derived_from" not in recorded
    ds.metadata()  # must serialise end to end


def test_file_without_provenance_still_loads(tmp_path) -> None:
    # Files written before 'derived_from' existed.
    path = write_file(tmp_path / "old.npz")
    f = ResidualStatisticsFile.load(path)
    assert f.provenance is None
    assert "anemoi_datasets_version" not in f.metadata()


# --------------------------------------------------------------------------- #
# The npz format
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_compute_writes_a_readable_npz_residual_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = compute_residual(tmp_path, name="residual.npz")

    # Everything that is not a statistic is a JSON string under 'metadata', and
    # each statistic is an array of its own.
    with np.load(path) as archive:
        metadata = json.loads(str(archive[METADATA_KEY]))
        assert metadata["kind"] == RESIDUAL_KIND
        assert metadata["version"] == VERSION
        assert metadata["datasets"] == [DS0, DS1]
        assert metadata["variables"] == VARIABLES
        assert "statistics" not in metadata
        for key in STATISTICS_KEYS:
            assert archive[f"statistics.{key}"].shape == (len(VARIABLES),)
        np.testing.assert_allclose(archive["statistics.mean"], -0.1, atol=1e-6)

    # ... and the archive can be attached to a dataset like the JSON file.
    ds = open_dataset(DS0, residual_statistics=path)
    np.testing.assert_allclose(ds.residual_statistics["mean"], -0.1, atol=1e-6)
    np.testing.assert_allclose(ds.residual_statistics["stdev"], 0.0, atol=1e-6)
    assert ds.metadata_specific()["residual_statistics"]["path"] == path


def test_npz_round_trip_keeps_nan_and_the_other_blocks(tmp_path) -> None:
    document = {
        "kind": RESIDUAL_KIND,
        "version": VERSION,
        "datasets": [DS0, DS1],
        "variables": VARIABLES,
        "tendency": "6h",
        "grid": None,
        "statistics": {k: np.array([0.0, np.nan, 2.0, 3.0]) for k in STATISTICS_KEYS},
        "tendency_statistics": {k: np.array([4.0, 5.0, 6.0, 7.0]) for k in STATISTICS_KEYS},
        "compare": {"a": {"mean": {"abs_diff": 0.5}}},
    }

    path = str(tmp_path / "r.npz")
    save(path, document)
    back = load_document(path)

    assert back["tendency"] == "6h"
    assert back["grid"] is None
    assert back["compare"] == {"a": {"mean": {"abs_diff": 0.5}}}
    for key in STATISTICS_KEYS:
        np.testing.assert_array_equal(back["statistics"][key], [0.0, np.nan, 2.0, 3.0])
        np.testing.assert_array_equal(back["tendency_statistics"][key], [4.0, 5.0, 6.0, 7.0])

    # NaN survives as NaN, and reaches the reader as such.
    stats = ResidualStatisticsFile.load(path).select(VARIABLES)
    assert np.isnan(stats["mean"][1])


def test_not_an_npz(tmp_path) -> None:
    path = tmp_path / "r.npz"
    path.write_text("not an archive at all")
    with pytest.raises(ValueError, match="not a valid numpy archive"):
        ResidualStatisticsFile.load(path)


def test_npz_without_metadata(tmp_path) -> None:
    path = str(tmp_path / "r.npz")
    np.savez(path, **{f"statistics.{k}": np.zeros(4) for k in STATISTICS_KEYS})
    with pytest.raises(ValueError, match="no 'metadata' entry"):
        ResidualStatisticsFile.load(path)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_metadata_specific_records_the_residual_statistics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = compute_residual(tmp_path)
    ds = open_dataset(DS0, residual_statistics=path)

    for md in (ds.metadata_specific(), ds.metadata()["specific"]):
        recorded = md["residual_statistics"]
        assert recorded["path"] == path
        assert recorded["kind"] == RESIDUAL_KIND
        assert recorded["version"] == VERSION
        assert recorded["datasets"] == [DS0, DS1]
        assert recorded["variables"] == ds.variables

    # The tree records where the statistics came from.
    assert path in str(ds.tree())


# --------------------------------------------------------------------------- #
# Absent residual statistics
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_no_residual_statistics_by_default() -> None:
    ds = open_dataset(DS0)
    assert not hasattr(ds, "residual_statistics")
    with pytest.raises(ResidualStatisticsNotAvailable, match="No residual statistics"):
        ds.residual_statistics


# --------------------------------------------------------------------------- #
# Propagation through the wrapping datasets
# --------------------------------------------------------------------------- #


@mockup_open_zarr
def test_select_after_residual_statistics(tmp_path, monkeypatch) -> None:
    """A `select` applied on top of the wrapper re-indexes the residuals."""
    monkeypatch.chdir(tmp_path)
    path = write_file(tmp_path / "r.npz")

    ds = open_dataset({"dataset": DS0, "residual_statistics": path}, select=["b", "d"])
    assert ds.variables == ["b", "d"]
    np.testing.assert_allclose(ds.residual_statistics["mean"], [1.0, 3.0])


@mockup_open_zarr
def test_select_before_residual_statistics(tmp_path, monkeypatch) -> None:
    """A `select` in the same call is applied first; the file is then subsetted."""
    monkeypatch.chdir(tmp_path)
    path = write_file(tmp_path / "r.npz")

    ds = open_dataset(DS0, select=["d", "b"], residual_statistics=path)
    assert ds.variables == ["d", "b"]
    np.testing.assert_allclose(ds.residual_statistics["mean"], [3.0, 1.0])


@mockup_open_zarr
def test_rename_keeps_the_positional_residuals(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = write_file(tmp_path / "r.npz")

    ds = open_dataset({"dataset": DS0, "residual_statistics": path}, rename={"a": "A"})
    assert ds.variables == ["A", "b", "c", "d"]
    np.testing.assert_allclose(ds.residual_statistics["mean"], [0.0, 1.0, 2.0, 3.0])


@mockup_open_zarr
def test_subset_by_dates_keeps_the_residuals(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = write_file(tmp_path / "r.npz")

    ds = open_dataset(DS0, residual_statistics=path, start=2021, end=2021, frequency="12h")
    np.testing.assert_allclose(ds.residual_statistics["mean"], [0.0, 1.0, 2.0, 3.0])


@mockup_open_zarr
def test_join_concatenates_the_residuals(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    left = write_file(tmp_path / "left.npz", variables=["a", "b"], values={k: [0.0, 1.0] for k in STATISTICS_KEYS})
    right = write_file(tmp_path / "right.npz", variables=["c", "d"], values={k: [2.0, 3.0] for k in STATISTICS_KEYS})

    ds = open_dataset(
        join=[
            {"dataset": DS0, "select": ["a", "b"], "residual_statistics": left},
            {"dataset": DS0, "select": ["c", "d"], "residual_statistics": right},
        ]
    )
    assert ds.variables == ["a", "b", "c", "d"]
    np.testing.assert_allclose(ds.residual_statistics["mean"], [0.0, 1.0, 2.0, 3.0])


@mockup_open_zarr
def test_rescale_scales_the_residuals(tmp_path, monkeypatch) -> None:
    """A residual is a difference, so the offset cancels and only the scale applies."""
    monkeypatch.chdir(tmp_path)
    path = write_file(
        tmp_path / "r.npz",
        values={k: [1.0, 1.0, 1.0, 1.0] for k in STATISTICS_KEYS},
    )

    ds = open_dataset({"dataset": DS0, "residual_statistics": path}, rescale={"a": (2.0, 100.0)})
    np.testing.assert_allclose(ds.residual_statistics["mean"], [2.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(ds.residual_statistics["stdev"], [2.0, 1.0, 1.0, 1.0])


def test_trajectories_dataset(tmp_path) -> None:
    """Trajectory datasets resolve their own layout-aware wrapper."""
    from test_trajectories import make_trajectories_zarr
    from test_trajectories import open_trajectories_zarr

    path = write_file(
        tmp_path / "r.npz", variables=["a", "b", "c"], values={k: [0.0, 1.0, 2.0] for k in STATISTICS_KEYS}
    )

    ds = open_trajectories_zarr(make_trajectories_zarr())._subset(residual_statistics=path).mutate()
    assert type(ds).__module__.endswith("trajectories.residual_statistics")
    np.testing.assert_allclose(ds.residual_statistics["mean"], [0.0, 1.0, 2.0])

    # The trajectory metadata keys survive the extra wrapper.
    md = ds.dataset_metadata()
    assert md["base_frequency"] == "6h"
    assert md["step_frequency"] == "6h"
    assert md["specific"]["residual_statistics"]["datasets"] == [DS0, DS1]
    ds.metadata()  # must serialise end to end


# --------------------------------------------------------------------------- #
# File validation
# --------------------------------------------------------------------------- #


def test_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ResidualStatisticsFile.load(tmp_path / "nope.npz")


def test_no_kind(tmp_path) -> None:
    path = write_file(tmp_path / "r.npz", kind=None)
    with pytest.raises(ValueError, match="not a residual-statistics file"):
        ResidualStatisticsFile.load(path)


def test_future_version(tmp_path) -> None:
    path = write_file(tmp_path / "r.npz", version=VERSION + 1)
    with pytest.raises(ValueError, match="newer than the supported version"):
        ResidualStatisticsFile.load(path)


@pytest.mark.parametrize("datasets", [[DS0], [DS0, DS1, DS0], [DS0, ""], [DS0, None]])
def test_needs_two_dataset_names(tmp_path, datasets) -> None:
    path = write_file(tmp_path / "r.npz", datasets=datasets)
    with pytest.raises(ValueError, match="'datasets' must be a list"):
        ResidualStatisticsFile.load(path)


def test_duplicated_variables(tmp_path) -> None:
    path = write_file(
        tmp_path / "r.npz",
        variables=["a", "a", "b"],
        values={k: [0.0, 1.0, 2.0] for k in STATISTICS_KEYS},
    )
    with pytest.raises(ValueError, match="duplicated variables"):
        ResidualStatisticsFile.load(path)


def test_missing_statistic(tmp_path) -> None:
    values = {k: [0.0] * len(VARIABLES) for k in STATISTICS_KEYS if k != "stdev"}
    path = write_file(tmp_path / "r.npz", values=values)
    with pytest.raises(ValueError, match=r"missing \['stdev'\]"):
        ResidualStatisticsFile.load(path)


def test_wrong_number_of_values(tmp_path) -> None:
    values = {k: [0.0] * len(VARIABLES) for k in STATISTICS_KEYS}
    values["mean"] = [0.0]
    path = write_file(tmp_path / "r.npz", values=values)
    with pytest.raises(ValueError, match="has 1 values but there are 4 variables"):
        ResidualStatisticsFile.load(path)


def test_null_in_the_metadata_means_nan(tmp_path) -> None:
    # The arrays hold NaN as NaN; 'null' is the JSON metadata's spelling of it,
    # and is still accepted for a statistic stored there.
    values = {k: [0.0] * len(VARIABLES) for k in STATISTICS_KEYS}
    values["stdev"] = [None] * len(VARIABLES)
    path = write_file(tmp_path / "r.npz", values=values)
    stats = ResidualStatisticsFile.load(path).select(VARIABLES)
    assert np.all(np.isnan(stats["stdev"]))


def test_file_may_hold_extra_variables(tmp_path) -> None:
    path = write_file(tmp_path / "r.npz")
    stats = ResidualStatisticsFile.load(path).select(["c", "a"])
    np.testing.assert_allclose(stats["mean"], [2.0, 0.0])


@mockup_open_zarr
def test_file_must_cover_every_variable(tmp_path) -> None:
    path = write_file(
        tmp_path / "r.npz",
        variables=["a", "b"],
        values={k: [0.0, 1.0] for k in STATISTICS_KEYS},
    )
    with pytest.raises(ValueError, match=r"no residual statistics for variable\(s\) \['c', 'd'\]"):
        open_dataset(DS0, residual_statistics=path)
