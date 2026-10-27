# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Monitoring plots of a gridded observation dataset, into <dataset>/_monitor/.

timeseries.png    per day and group (e.g. satellite): cells filled, mean ± std per variable
                  (the ``obs_stats`` diagnostics)
qc.png            per day and group: what each QC step removes, and the cells filled
                  (``obs_qc`` and ``obs_stats``)
maps_YYYY-MM.png  one month per cell: steps filled, QC rejection (``obs_qc_cells``), mean /
                  std of the map variables and one step of them (read from the dataset)

The diagnostics come from ``anemoi.datasets.create.diagnostics``, written by sources such
as ``grid_observations``; plots whose inputs are missing are skipped.
"""

import json
import logging
import os
from functools import cache

import numpy as np
import pandas as pd

from anemoi.datasets.create.diagnostics import FOLDER

LOG = logging.getLogger(__name__)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def daily(stats: pd.DataFrame) -> pd.DataFrame:
    """Merge obs_stats rows per (day, group, column), with mean and std."""
    out = stats.groupby(["day", "group", "column"]).agg(
        count=("count", "sum"),
        nan_count=("nan_count", "sum"),
        sum=("sum", "sum"),
        sum_sq=("sum_sq", "sum"),
    )
    out["mean"] = out["sum"] / out["count"]
    out["std"] = np.sqrt(np.maximum(out["sum_sq"] / out["count"] - out["mean"] ** 2, 0))
    return out.reset_index()


def plot_timeseries(stats: pd.DataFrame, out: str, title: str = "") -> str:
    """Daily cells filled per group, and daily mean ± std per variable per group."""
    plt = _plt()
    d = daily(stats)
    columns = list(dict.fromkeys(stats["column"]))
    fig, axes = plt.subplots(1 + len(columns), 1, figsize=(12, 2.2 * (1 + len(columns))), sharex=True)
    d[d["column"] == columns[0]].pivot(index="day", columns="group", values="count").plot(
        ax=axes[0], marker=".", title="cells filled per day"
    )
    for ax, col in zip(axes[1:], columns):
        for group, one in d[d["column"] == col].groupby("group"):
            ax.plot(one["day"], one["mean"], ".-", label=str(group))
            ax.fill_between(one["day"], one["mean"] - one["std"], one["mean"] + one["std"], alpha=0.15)
        ax.set_title(col)
    axes[0].legend(title="group", ncols=8, fontsize="small")
    fig.suptitle(title)
    return _save(fig, out)


def plot_qc(qc: pd.DataFrame, stats: pd.DataFrame, n_cells_per_day: int, out: str, title: str = "") -> str:
    """Rows in, fraction removed by each QC step and the cell coverage, per day and group."""
    plt = _plt()
    qc = qc.groupby(["day", "group", "step"])["n"].sum().unstack("step").fillna(0)
    rows_in = qc["in"]
    steps = [s for s in qc.columns if s not in ("in", "kept") and not s.startswith("range:")]
    ranges = [s for s in qc.columns if s.startswith("range:")]
    d = daily(stats)
    cover = d[d["column"] == d["column"].iloc[0]].pivot(index="day", columns="group", values="count")
    panels = 3 + len(steps) + bool(ranges)
    fig, axes = plt.subplots(panels, 1, figsize=(12, 2.4 * panels), sharex=True)
    rows_in.unstack("group").plot(ax=axes[0], marker=".", title="rows read per day")
    (qc["kept"] / rows_in).unstack("group").plot(ax=axes[1], marker=".", title="fraction of rows kept (selectable)")
    for ax, step in zip(axes[2:], steps):
        (qc[step] / rows_in).unstack("group").plot(ax=ax, marker=".", title=f"fraction {step}")
    if ranges:  # values masked per column, every group
        pooled = qc[ranges].groupby("day").sum().div(rows_in.groupby("day").sum(), axis=0)
        pooled.plot(ax=axes[2 + len(steps)], legend=len(ranges) <= 12, title="fraction of values out of range")
    total = cover.sum(axis=1).rename("all")
    (cover.join(total) / n_cells_per_day).plot(ax=axes[-1], marker=".", title="fraction of cells filled")
    for ax in axes:
        ax.set_xlabel("")
    fig.suptitle(title)
    return _save(fig, out)


@cache
def _rows(latitudes: bytes) -> tuple | None:
    """Row sizes when the points run in latitude rows (each row's points together)."""
    lat = np.frombuffer(latitudes)
    starts = np.flatnonzero(np.r_[True, lat[1:] != lat[:-1]])
    if np.unique(lat[starts]).size != starts.size:
        return None
    return tuple(np.diff(np.r_[starts, lat.size]))


def cell_boxes(lat: np.ndarray, lon: np.ndarray) -> np.ndarray | None:
    """(cell, 4, 2) lon/lat corners of each cell for a grid in latitude rows (reduced
    Gaussian, regular): rows meet halfway between their latitudes, a point spans its row's
    longitude step. None for other grids."""
    rows = _rows(np.ascontiguousarray(lat, np.float64).tobytes())
    if rows is None:
        return None
    rows = np.array(rows)
    row_lat = lat[np.r_[0, np.cumsum(rows)[:-1]]]
    edges = np.r_[90.0, (row_lat[1:] + row_lat[:-1]) / 2, -90.0]
    top, bottom = np.repeat(edges[:-1], rows), np.repeat(edges[1:], rows)
    half = np.repeat(180.0 / rows, rows)
    lon = np.where(lon > 180, lon - 360, lon)
    west, east = lon - half, lon + half
    return np.stack([np.column_stack(c) for c in ((west, top), (east, top), (east, bottom), (west, bottom))], axis=1)


def draw_cells(ax, lat, lon, values, vmin=None, vmax=None, cmap=None):
    """Each cell's value as its box (NaN: blank), or as dots on grids not in rows."""
    from matplotlib.collections import PolyCollection

    boxes = cell_boxes(lat, lon)
    if boxes is None:
        lon = np.where(lon > 180, lon - 360, lon)
        im = ax.scatter(lon, lat, c=values, s=1, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        wrap = boxes[:, 1, 0] > 180
        boxes = np.concatenate([boxes, boxes[wrap] - [360.0, 0.0]])
        values = np.concatenate([values, values[wrap]])
        im = PolyCollection(boxes, cmap=cmap, linewidths=0, antialiased=False)
        im.set_array(np.ma.masked_invalid(values))
        im.set_clim(vmin, vmax)
        ax.add_collection(im)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    return im


def plot_maps(path: str, month: str, variables: list[str], out: str, qc_cells: np.ndarray | None = None) -> str:
    """Steps filled and QC rejection per cell over one month, and the mean / std / one
    mid-month step of each variable."""
    from anemoi.datasets import open_dataset

    plt = _plt()
    start = pd.Timestamp(month)
    end = start + pd.offsets.MonthEnd(1) + pd.Timedelta("23h59min")
    ds = open_dataset(path, select=variables, start=start.to_pydatetime(), end=end.to_pydatetime())
    lat, lon = ds.latitudes, ds.longitudes
    data = ds[:, :, 0, :].astype(np.float64)  # (date, variable, cell)
    count = np.isfinite(data).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(data, 0)
        std = np.nanstd(data, 0)
    middle = len(ds.dates) // 2

    def show(ax, values, title, **kw):
        im = draw_cells(ax, lat, lon, values, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig, axes = plt.subplots(1 + len(variables), 3, figsize=(15, 3.2 * (1 + len(variables))))
    axes = np.atleast_2d(axes)
    show(axes[0, 0], count.max(0) / len(ds.dates), "fraction of steps filled", vmin=0)
    if qc_cells is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            show(axes[0, 1], qc_cells[1] / qc_cells[0], "fraction of rows rejected by QC", vmin=0)
    else:
        axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    for k, name in enumerate(ds.variables):
        lo, hi = np.nanpercentile(mean[k], [1, 99]) if np.isfinite(mean[k]).any() else (0, 1)
        show(axes[k + 1, 0], mean[k], f"{name}: mean", vmin=lo, vmax=hi)
        show(axes[k + 1, 1], std[k], f"{name}: std")
        show(axes[k + 1, 2], data[middle, k], f"{name}: {pd.Timestamp(ds.dates[middle]):%Y-%m-%d %HZ}")
    fig.suptitle(f"{os.path.basename(path.rstrip('/'))} {month}")
    return _save(fig, out, rect=(0, 0, 1, 0.98))


def _save(fig, out: str, rect=None) -> str:
    fig.tight_layout(rect=rect)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=80)
    _plt().close(fig)
    return out


def _recipe_maps(path: str) -> list[str]:
    """The `maps` of the dataset's grid_observations source, if any."""
    import zarr

    attrs = zarr.open(path, mode="r").attrs
    recipe = attrs.get("recipe") or attrs.get("_recipe") or {}
    if isinstance(recipe, str):
        recipe = json.loads(recipe)

    def find(x):
        if isinstance(x, dict):
            if isinstance(x.get("grid_observations"), dict):
                return x["grid_observations"].get("maps", [])
            return next((m for v in x.values() if (m := find(v)) is not None), None)
        if isinstance(x, list):
            return next((m for v in x if (m := find(v)) is not None), None)
        return None

    return find(recipe) or []


def plot_all(path: str, month: str | None = None, variables: list[str] | None = None) -> list[str]:
    """Every plot whose inputs exist; maps for `month` (YYYY-MM), else the last month."""
    from anemoi.datasets import open_dataset

    folder, data = os.path.join(path, FOLDER), os.path.join(path, FOLDER, "data")
    name = os.path.basename(path.rstrip("/"))
    outs = []
    stats_file, qc_file = os.path.join(data, "obs_stats.parquet"), os.path.join(data, "obs_qc.parquet")
    stats = pd.read_parquet(stats_file) if os.path.exists(stats_file) else None
    if stats is not None:
        outs.append(plot_timeseries(stats, os.path.join(folder, "timeseries.png"), name))
    if stats is not None and os.path.exists(qc_file):
        ds = open_dataset(path)
        per_day = ds.shape[-1] * int(pd.Timedelta("1D") / pd.Timedelta(ds.frequency))
        outs.append(plot_qc(pd.read_parquet(qc_file), stats, per_day, os.path.join(folder, "qc.png"), name))
    variables = variables if variables is not None else _recipe_maps(path)
    if variables:
        if month is None:
            month = f"{pd.Timestamp(open_dataset(path).dates[-1]):%Y-%m}"
        qc_cells = None
        cells_file = os.path.join(data, "obs_qc_cells.npz")
        if os.path.exists(cells_file):
            with np.load(cells_file) as f:
                qc_cells = f.get(month)
        outs.append(plot_maps(path, month, variables, os.path.join(folder, f"maps_{month}.png"), qc_cells))
    for out in outs:
        LOG.info(f"Wrote {out}")
    return outs
