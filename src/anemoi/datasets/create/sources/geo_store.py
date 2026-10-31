# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os
import re

import numpy as np
import pandas as pd

from ..source import Source
from . import source_registry

LOG = logging.getLogger(__name__)

NO_CHUNK = 2**64 - 1  # a sharded chunk never written (zarr 3 sharding index)
DTYPES = {"uint8": "<u1", "uint16": "<u2", "int32": "<i4", "int64": "<i8", "float32": "<f4", "float64": "<f8"}
BYTES = {"name": "bytes", "configuration": {"endian": "little"}}


def _zstd(data: bytes) -> bytes:
    from numcodecs import Zstd

    return Zstd().decode(data)


class ZarrArray:
    """One array of a zarr 3 store, in the two layouts obsbase writes (obsbase.geo.store):
    one chunk of the whole array (bytes + zstd), or one shard of the whole array holding
    one zstd chunk per row of the first axis (sharding_indexed, index at the end). Any other
    layout is an error: this reads obsbase's stores without needing zarr 3.
    """

    def __init__(self, path: str):
        with open(os.path.join(path, "zarr.json")) as f:
            meta = json.load(f)
        if meta.get("zarr_format") != 3 or meta.get("node_type") != "array":
            raise ValueError(f"{path}: not a zarr 3 array")
        if meta["data_type"] not in DTYPES or meta["chunk_key_encoding"]["configuration"]["separator"] != "/":
            raise NotImplementedError(f"{path}: data type {meta['data_type']} or chunk key encoding not read")
        self.path = path
        self.shape = tuple(meta["shape"])
        self.dtype = np.dtype(DTYPES[meta["data_type"]])
        self.fill = meta["fill_value"]
        self.attrs = meta.get("attributes", {})
        self.dims = meta.get("dimension_names")
        if list(meta["chunk_grid"]["configuration"]["chunk_shape"]) != list(self.shape):
            raise NotImplementedError(f"{path}: more than one chunk (or shard)")
        self.file = os.path.join(path, "c", *["0"] * len(self.shape))
        codecs = meta["codecs"]
        if [c["name"] for c in codecs] == ["bytes", "zstd"] and codecs[0] == BYTES:
            self.inner = None
        elif len(codecs) == 1 and codecs[0]["name"] == "sharding_indexed":
            shard = codecs[0]["configuration"]
            inner, chunk = shard["codecs"], shard["chunk_shape"]
            index = [c["name"] for c in shard["index_codecs"]]
            if [c["name"] for c in inner] != ["bytes", "zstd"] or inner[0] != BYTES or index != ["bytes", "crc32c"]:
                raise NotImplementedError(f"{path}: shard codecs not read")
            if shard["index_location"] != "end" or chunk[0] != 1 or list(chunk[1:]) != list(self.shape[1:]):
                raise NotImplementedError(f"{path}: shard layout not read (chunks must be single rows)")
            self.inner = chunk
        else:
            raise NotImplementedError(f"{path}: codecs {[c['name'] for c in codecs]} not read")

    def read(self, rows: list[int] | None = None) -> np.ndarray:
        """The whole array, or the given rows of the first axis."""
        if self.inner is None:
            with open(self.file, "rb") as f:
                out = np.frombuffer(_zstd(f.read()), self.dtype).reshape(self.shape)
            return out if rows is None else out[rows]
        rows = list(range(self.shape[0])) if rows is None else list(rows)
        out = np.full((len(rows), *self.shape[1:]), self.fill, self.dtype)
        with open(self.file, "rb") as f:
            f.seek(-(16 * self.shape[0] + 4), os.SEEK_END)
            index = np.frombuffer(f.read(16 * self.shape[0]), "<u8").reshape(-1, 2)
            for i, row in enumerate(rows):
                offset, size = (int(x) for x in index[row])
                if offset == NO_CHUNK and size == NO_CHUNK:
                    continue
                f.seek(offset)
                out[i] = np.frombuffer(_zstd(f.read(size)), self.dtype).reshape(self.shape[1:])
        return out

    def decoded(self, rows: list[int] | None = None) -> np.ndarray:
        """float32 values: raw * scale_factor + add_offset, NaN at _FillValue."""
        raw = self.read(rows)
        out = raw.astype(np.float32) * np.float32(self.attrs.get("scale_factor", 1.0))
        out += np.float32(self.attrs.get("add_offset", 0.0))
        if "_FillValue" in self.attrs:
            out[raw == self.attrs["_FillValue"]] = np.nan
        return out


def store_times(path: str) -> pd.DatetimeIndex:
    """The time axis of a day store (CF units "<unit> since <time>")."""
    time = ZarrArray(os.path.join(path, "time"))
    match = re.fullmatch(r"(\w+) since (.+)", time.attrs["units"])
    if not match:
        raise ValueError(f"{path}: time units {time.attrs['units']!r}")
    return pd.Timestamp(match[2]) + pd.to_timedelta(time.read(), unit=match[1])


@source_registry.register("geo_store")
class GeoStoreSource(Source):
    """Observation rows from obsbase's geo day stores (zarr 3: the geostationary slot
    streams and GridSat): one row per (hour, cell) holding a value, at the cell's centre
    and the hour's nominal time, as a DataFrame for ``grid_observations``.

    Only the hours inside some window of the dates served are read (``TimeSpan.window``).
    A value is kept only where its own satellite (the variable's ``satid`` attribute, e.g.
    ``satid_wv``) is the row's (``satid``, e.g. ``satid_ir``): a row never mixes satellites.

    Examples
    --------
    .. code-block:: yaml

      geo_store:
        path: /ec/project/dop/base_data/gridsat_noaa_cdr/year=%Y/month=%m/%Y-%m-%d.zarr
        satid: satid_ir

    Parameters
    ----------
    context : Any
        The build context.
    path : str
        A strftime template of one day's store.
    satid : str
        The store variable giving each row's satellite (column ``satid``).
    columns : list[str], optional
        Store variables to read on top of those a calling source asks for.
    missing_files : str
        "error" (default) or "skip": a day without a store has no rows.
    """

    emoji = "🛰️"  # For tracing

    def __init__(self, context, path: str, satid: str, columns: list[str] | None = None, missing_files="error"):
        super().__init__(context)
        if missing_files not in ("error", "skip"):
            raise ValueError(f"geo_store: missing_files must be 'error' or 'skip', not {missing_files!r}")
        self.path = path
        self.satid = satid
        self.columns = list(columns or [])
        self.missing_files = missing_files

    def execute_time_span(self, span) -> pd.DataFrame:
        names = [
            c
            for c in dict.fromkeys([*(span.columns or []), *self.columns])
            if c not in ("date", "latitude", "longitude", "satid")
        ]
        if not names:
            raise ValueError("geo_store: no variable asked for (set columns or call it from grid_observations)")
        days = pd.date_range(pd.Timestamp(span.start).floor("D"), pd.Timestamp(span.end).floor("D"))
        frames = []
        for day in days:
            path = day.strftime(self.path)
            if not os.path.exists(path):
                if self.missing_files == "error":
                    raise FileNotFoundError(f"geo_store: no day store {path}")
                continue
            frames.append(self.read_day(path, span, names))
        self.context.trace(self.emoji, f"geo_store {span}: {len(frames)} days, {sum(map(len, frames))} rows")
        columns = ["date", "latitude", "longitude", "satid", *names]
        if not frames:
            return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "date" else float) for c in columns})
        return pd.concat(frames, ignore_index=True)[columns]

    def read_day(self, path: str, span, names: list[str]) -> pd.DataFrame:
        from anemoi.transform.obs.grid import grid_points

        with open(os.path.join(path, "zarr.json")) as f:
            grid = json.load(f)["attributes"]["grid"]
        times = store_times(path)
        hours = np.flatnonzero(np.asarray(span.in_windows(times)))
        if hours.size == 0:
            return pd.DataFrame()
        cells = ZarrArray(os.path.join(path, "cell")).read()
        satid = self._read(path, self.satid, hours, cells.size)
        values = {}
        for name in names:
            array = ZarrArray(os.path.join(path, name))
            if array.dims is not None and array.dims[0] != "time":
                raise ValueError(f"geo_store: {name} is not a per-hour variable ({array.dims})")
            x = array.decoded(list(hours))
            own = array.attrs.get("satid")
            if own and own != self.satid:
                x[self._read(path, own, hours, cells.size) != satid] = np.nan
            values[name] = x
        present = np.zeros((hours.size, cells.size), bool)
        for x in values.values():
            present |= np.isfinite(x)
        if (present & np.isnan(satid)).any():
            raise ValueError(f"geo_store: {path} has values without a satellite ({self.satid})")
        hour, cell = np.nonzero(present)
        lat, lon = grid_points(grid)
        rows = {
            "date": times[hours][hour],
            "latitude": lat[cells[cell]],
            "longitude": lon[cells[cell]],
            "satid": satid[hour, cell].astype(np.int64),
        }
        return pd.DataFrame(rows | {name: x[hour, cell] for name, x in values.items()})

    @staticmethod
    def _read(path: str, name: str, hours: np.ndarray, n_cells: int) -> np.ndarray:
        """A satellite variable as float (hour, cell): per (hour, cell), or per hour (slots)."""
        array = ZarrArray(os.path.join(path, name))
        x = array.decoded(list(hours))
        return np.broadcast_to(x[:, None], (hours.size, n_cells)) if x.ndim == 1 else x
