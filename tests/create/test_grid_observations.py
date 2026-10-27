# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from anemoi.utils.window import Window

from anemoi.datasets import open_dataset

from .utils.create import create_dataset

pytest.importorskip("pyarrow")
pytest.importorskip("anemoi.transform.obs")

CELL = 500


def write_days(root, rows):
    """Per-day parquet files of obsbase's layout (datetime, lat, lon, ...)."""
    from anemoi.transform.obs.grid import grid_points

    lat, lon = grid_points("o96")
    df = pd.DataFrame(rows, columns=["datetime", "satid", "bt_1"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["lat"], df["lon"] = lat[CELL], lon[CELL]
    for day, part in df.groupby(df["datetime"].dt.floor("D")):
        path = root / f"{day:%Y-%m-%d}.parquet"
        part.to_parquet(path, index=False)
    return str(root / "%Y-%m-%d.parquet")


def recipe(path, **parquet):
    return {
        "dates": {"start": "2020-01-01T00:00:00", "end": "2020-01-02T18:00:00", "frequency": "6h"},
        "build": {"group_by": "daily", "allow_nans": True},
        "input": {
            "grid_observations": {
                "window": "(-350m,+10m]",
                "grid": "o96",
                "select": {"candidates": "latest_pass", "reduce": "nearest"},
                "qc": {"range": {"bt_1": [100, 300]}},
                "variables": {
                    "channels": ["bt_1"],
                    "columns": ["satid"],
                    "derived": ["time_delta", "n_obs"],
                    "forcings": ["cos_latitude"],
                },
                "input": {
                    "parquet": {"path": path, "flavour": {"date": "datetime", "latitude": "lat", "longitude": "lon"}}
                    | parquet
                },
            }
        },
    }


def test_grid_observations_windows_cross_the_day(tmp_path):
    rows = [
        ("2020-01-01 06:10", 3, 250.0),  # T = 06Z (the window end is included)
        ("2020-01-01 23:00", 5, 260.0),  # T = 2020-01-02 00Z: read by the next day's group
        ("2020-01-02 11:00", 3, 400.0),  # out of range: masked, then incomplete
    ]
    path = write_days(tmp_path, rows)
    ds = open_dataset(create_dataset(recipe=recipe(path, missing_files="skip"), output=None))
    assert ds.shape == (8, 5, 1, 40320)
    assert ds.variables == sorted(["satid", "bt_1", "time_delta", "n_obs", "cos_latitude"])
    v = ds.name_to_index
    bt = ds[:, v["bt_1"], 0, CELL]
    assert bt[1] == 250 and bt[4] == 260 and np.isnan(bt[[0, 2, 3, 5, 6, 7]]).all()
    assert ds[4, v["satid"], 0, CELL] == 5
    assert ds[4, v["time_delta"], 0, CELL] == pytest.approx(-1.0)
    assert np.isnan(ds[:, v["bt_1"], 0, :CELL]).all()
    assert np.isfinite(ds[:, v["cos_latitude"], 0, :]).all()
    assert ds.statistics["mean"][v["bt_1"]] == pytest.approx(255)


def test_parquet_time_span_ends_and_missing_files(tmp_path):
    from anemoi.datasets.create.arguments import TimeSpan
    from anemoi.datasets.create.sources.parquet import ParquetSource

    class Context:
        def trace(self, *args):
            pass

    path = write_days(tmp_path, [("2020-01-01 06:00", 3, 250.0), ("2020-01-01 12:00", 3, 251.0)])
    source = ParquetSource(Context(), path, flavour={"date": "datetime", "latitude": "lat", "longitude": "lon"})
    t0, t1 = datetime.datetime(2020, 1, 1, 6), datetime.datetime(2020, 1, 1, 12)
    assert source.execute(TimeSpan(t0, t1, "right"))["bt_1"].tolist() == [251.0]
    assert source.execute(TimeSpan(t0, t1, "left"))["bt_1"].tolist() == [250.0]
    both = source.execute(TimeSpan(t0, t1, "both", columns=["satid"]))
    assert list(both.columns) == ["date", "latitude", "longitude", "satid"] and len(both) == 2
    with pytest.raises(FileNotFoundError, match="missing"):
        source.execute(TimeSpan(t0, t1 + datetime.timedelta(days=1)))
    skip = ParquetSource(Context(), path, missing_files="skip", flavour=source.flavour)
    assert len(skip.execute(TimeSpan(t0, t1 + datetime.timedelta(days=1)))) == 1


def test_diagnostics_are_merged_under_monitor(tmp_path):
    rows = [("2020-01-01 06:10", 3, 250.0), ("2020-01-02 11:00", 3, 400.0)]
    (tmp_path / "base").mkdir()
    path = write_days(tmp_path / "base", rows)
    created = create_dataset(recipe=recipe(path, missing_files="skip"), output=str(tmp_path / "ds.zarr"))
    data = tmp_path / "ds.zarr" / "_monitor" / "data"
    stats = pd.read_parquet(data / "obs_stats.parquet")
    assert set(stats["column"]) == {"satid", "bt_1", "time_delta", "n_obs"}
    qc = pd.read_parquet(data / "obs_qc.parquet").set_index(["day", "step"])["n"]
    assert qc[pd.Timestamp("2020-01-02"), "range:bt_1"] == 1
    with np.load(data / "obs_qc_cells.npz") as cells:
        assert list(cells) == ["2020-01"] and cells["2020-01"][0, CELL] == 2  # rows read, each by one group
    assert open_dataset(created).shape == (8, 5, 1, 40320)


def test_monitor_plots(tmp_path):
    pytest.importorskip("matplotlib")
    from anemoi.datasets.monitor import plot_all

    (tmp_path / "base").mkdir()
    path = write_days(tmp_path / "base", [("2020-01-01 06:10", 3, 250.0), ("2020-01-02 00:00", 5, 260.0)])
    config = recipe(path, missing_files="skip")
    config["input"]["grid_observations"]["maps"] = ["bt_1", "n_obs"]
    created = create_dataset(recipe=config, output=str(tmp_path / "ds.zarr"))
    outs = plot_all(created)
    assert [os.path.basename(o) for o in outs] == ["timeseries.png", "qc.png", "maps_2020-01.png"]
    assert all(os.path.getsize(o) > 0 for o in outs)


@pytest.mark.skipif(shutil.which("odc") is None, reason="needs the odc command")
def test_odb_time_span_per_cycle_files_with_milliseconds(tmp_path):
    pyodc = pytest.importorskip("pyodc")
    from anemoi.datasets.create.arguments import TimeSpan
    from anemoi.datasets.create.sources.odb import OdbSource

    class Context:
        def trace(self, *args):
            pass

    # cycle 06Z holds (00:10, 06:10]; one row at 00:10:00.500, just inside it
    rows = {
        "date": [20200101] * 3,
        "time": [1000, 30000, 61001],  # 06:10:01: after the window
        "time_ms": [500.0, 0.0, 0.0],
        "lat": [10.0, 20.0, 30.0],
        "lon": [1.0, 2.0, 3.0],
        "bt_1": [250.0, 251.0, 252.0],
    }
    pyodc.encode_odb(pd.DataFrame(rows), str(tmp_path / "2020010106.odb"))
    flavour = {"date_column_name": "date", "time_column_name": "time", "latitude_column_name": "lat"}
    flavour |= {"longitude_column_name": "lon"}
    T = datetime.datetime(2020, 1, 1, 6)
    span = TimeSpan.from_window(Window("(-350m,+10m]"), [T])
    path = str(tmp_path / "{date:strftime(%Y%m%d%H)}.odb")
    exact = OdbSource(Context(), path, select="*", flavour=flavour | {"milliseconds_column_name": "time_ms"})
    df = exact.execute(span)
    assert df["bt_1"].tolist() == [250.0, 251.0] and "time_ms" not in df
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-01 00:10:00.500")
    seconds = OdbSource(Context(), path, select="*", flavour=flavour).execute(span)
    assert seconds["bt_1"].tolist() == [251.0]  # 00:10:00 is outside the window
