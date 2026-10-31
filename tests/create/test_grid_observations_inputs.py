# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# grid_observations with several inputs, and the geo_store source (obsbase geo day stores).

import datetime
import os

import numpy as np
import pandas as pd
import pytest
from anemoi.utils.window import Window

from anemoi.datasets import open_dataset

from .utils.create import create_dataset

pytest.importorskip("pyarrow")
pytest.importorskip("anemoi.transform.obs")

CELL = 500
# A two-cell (o96 500, 501) GridSat-like day store written by zarr 3 as obsbase does
# (sharded, zstd): irwin_cdr 250 + 2 K per 3 h at cell 500, irwvp 230 + ..., satid 270; at
# 06Z the WV pixel is from satellite 271; at 12Z cell 501 has WV only; 21Z never written.
GEO = os.path.join(os.path.dirname(__file__), "data", "geo_store", "%Y-%m-%d.zarr")
DATES = {"start": "2020-01-01T00:00:00", "end": "2020-01-01T18:00:00", "frequency": "6h"}


class Context:
    def trace(self, *args):
        pass


def test_geo_store_rows_inside_the_windows():
    from anemoi.transform.obs.grid import grid_points

    from anemoi.datasets.create.arguments import TimeSpan
    from anemoi.datasets.create.sources.geo_store import GeoStoreSource

    dates = [datetime.datetime(2020, 1, 1, h) for h in (0, 6, 12, 18)]
    span = TimeSpan.from_window(Window("(-90m,+1m]"), dates, ["satid", "irwin_cdr", "irwvp"])
    df = GeoStoreSource(Context(), GEO, satid="satid_ir", missing_files="skip").execute(span)  # 00Z reads 2019-12-31
    assert df["date"].dt.hour.tolist() == [0, 6, 12, 12, 18]  # 03, 09, 15Z are between windows
    assert df["irwin_cdr"].tolist()[:3] == pytest.approx([250, 252, 254]) and np.isnan(df["irwin_cdr"][3])
    assert np.isnan(df["irwvp"][1]) and df["irwvp"][3] == pytest.approx(240)  # 06Z: another satellite
    assert (df["satid"] == 270).all()
    lat, lon = grid_points("o96")
    assert df["latitude"][0] == lat[CELL] and df["longitude"][3] == lon[CELL + 1]
    with pytest.raises(FileNotFoundError, match="no day store"):
        GeoStoreSource(Context(), GEO, satid="satid_ir").execute(
            TimeSpan.from_window(Window("(-90m,+1m]"), [dates[0] + datetime.timedelta(days=1)], ["irwin_cdr"])
        )


def test_zarr_reader_refuses_other_layouts(tmp_path):
    from anemoi.datasets.create.sources.geo_store import ZarrArray

    store = os.path.dirname(GEO.replace("%Y-%m-%d", "2020-01-01"))
    meta = open(os.path.join(store, "2020-01-01.zarr", "irwin_cdr", "zarr.json")).read()
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "zarr.json").write_text(meta.replace('"zstd"', '"blosc"'))
    with pytest.raises(NotImplementedError, match="codecs"):
        ZarrArray(str(tmp_path / "x"))
    assert ZarrArray(os.path.join(store, "2020-01-01.zarr", "irwin_cdr")).read([7]).tolist() == [
        [0, 0]
    ]  # zarr's fill_value


def test_geo_store_through_grid_observations():
    recipe = {
        "dates": DATES,
        "build": {"group_by": "daily", "allow_nans": True},
        "input": {
            "grid_observations": {
                "window": "(-90m,+1m]",
                "grid": "o96",
                "select": {"candidates": "latest_pass", "reduce": "mean"},
                "qc": {"range": {"irwin_cdr": [170, 345]}, "require_complete": "any"},
                "variables": {
                    "channels": ["irwin_cdr", "irwvp"],
                    "columns": ["satid"],
                    "derived": ["n_obs"],
                    "spread": ["irwin_cdr"],
                    "rename": {"satid": "satellite_id"},
                    "prefix": "gridsat_",
                },
                "inputs": [
                    {
                        "input": {"geo_store": {"path": GEO, "satid": "satid_ir", "missing_files": "skip"}},
                        "group_offset": 10000,
                    }
                ],
            }
        },
    }
    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    v = ds.name_to_index
    ir, wv = ds[:, v["gridsat_irwin_cdr"], 0, CELL], ds[:, v["gridsat_irwvp"], 0, CELL]
    assert ir.tolist() == pytest.approx([250, 252, 254, 256]) and np.isnan(wv[1]) and wv[2] == pytest.approx(234)
    assert (ds[:, v["gridsat_satellite_id"], 0, CELL] == 10270).all()
    assert ds[2, v["gridsat_irwvp"], 0, CELL + 1] == pytest.approx(240) and np.isnan(
        ds[2, v["gridsat_irwin_cdr"], 0, CELL + 1]
    )
    assert ds[0, v["gridsat_irwin_cdr_std"], 0, CELL] == 0


def write_days(root, rows):
    from anemoi.transform.obs.grid import grid_points

    lat, lon = grid_points("o96")
    df = pd.DataFrame(rows, columns=["datetime", "satid", "fov", "bt_1", "bt_2"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["lat"], df["lon"] = lat[CELL], lon[CELL]
    for day, part in df.groupby(df["datetime"].dt.floor("D")):
        part.to_parquet(root / f"{day:%Y-%m-%d}.parquet", index=False)
    return str(root / "%Y-%m-%d.parquet")


def test_inputs_split_one_stream_into_slots_and_skip_inputs_outside_their_dates(tmp_path):
    # satid 201 (an old instrument version) has another band in bt_2: its own slot bt2_2
    rows = [("2020-01-01 05:00", 201, 0, 250.0, 260.0), ("2020-01-01 11:00", 206, 55, 251.0, 261.0)]
    path = write_days(tmp_path, rows)
    flavour = {"date": "datetime", "latitude": "lat", "longitude": "lon"}
    source = {"parquet": {"path": path, "flavour": flavour, "missing_files": "skip"}}
    scan = {"n_positions": 56, "first_side": "left"}
    recipe = {
        "dates": DATES,
        "build": {"group_by": "daily", "allow_nans": True},
        "input": {
            "grid_observations": {
                "window": "(-350m,+10m]",
                "grid": "o96",
                "select": {"candidates": "latest_pass", "reduce": "mean"},
                "qc": {"require_complete": "any"},
                "variables": {
                    "channels": ["bt_1", "bt_2", "bt2_2"],
                    "columns": ["satid"],
                    "derived": ["scan_pos"],
                    "scan": "input",
                },
                "inputs": [
                    {
                        "input": source,
                        "groups": [201],
                        "channels": {"bt_1": "bt_1", "bt_2": "bt2_2"},
                        "group_offset": 10000,
                        "scan": scan,
                    },
                    {
                        "input": source,
                        "groups": [206],
                        "channels": ["bt_1", "bt_2"],
                        "group_offset": 10000,
                        "scan": scan,
                    },
                    # never read: its dates miss the dataset (the path does not exist)
                    {
                        "input": {"parquet": {"path": str(tmp_path / "none-%Y.parquet")}},
                        "dates": ["2021-01-01", None],
                        "scan": scan,
                    },
                ],
            }
        },
    }
    ds = open_dataset(create_dataset(recipe=recipe, output=None))
    v = ds.name_to_index
    x = {name: ds[:, v[name], 0, CELL] for name in ("bt_1", "bt_2", "bt2_2", "satid", "scan_pos")}
    assert x["bt_1"][1] == 250 and np.isnan(x["bt_2"][1]) and x["bt2_2"][1] == 260 and x["satid"][1] == 10201
    assert x["bt_2"][2] == 261 and np.isnan(x["bt2_2"][2]) and x["satid"][2] == 10206
    assert x["scan_pos"][1] == -1 and x["scan_pos"][2] == 1


def test_one_of_input_and_inputs():
    from anemoi.datasets.create.sources.grid_observations import GridObservationsSource

    cfg = {"window": "(-350m,+10m]", "frequency": "6h", "grid": "o96"}
    cfg |= {"select": {"candidates": "window", "reduce": "nearest"}, "variables": {"channels": ["bt_1"]}}
    with pytest.raises(ValueError, match="one of input and inputs"):
        GridObservationsSource(None, **cfg)
