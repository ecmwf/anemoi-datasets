# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import numpy as np
import pandas as pd

from ..source import Source
from . import source_registry
from .patterns import _expand

LOG = logging.getLogger(__name__)


@source_registry.register("netcdf-tabular")
class NetCDFTabularSource(Source):
    """A tabular source that reads point observations from NetCDF files.

    Each file is expected to hold one-dimensional variables that share a
    single "point" dimension, with one latitude, one longitude and one time
    value per point (e.g. satellite radiances or in-situ observations).
    The files are decoded following the CF conventions (``scale_factor``,
    ``add_offset``, ``_FillValue`` and time ``units``), so fill values become
    NaN and times become ``datetime64``, even if every file uses a different
    time reference.

    Only the files whose observation times overlap the requested date range
    are read; the selection uses the actual times in the files, not their
    names.
    """

    emoji = "📄"  # For tracing

    def __init__(
        self,
        context: Any,
        path: str | list[str],
        variables: list[str] | None = None,
        flavour: dict | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialise the NetCDFTabularSource.

        Parameters
        ----------
        context : Any
            The context for the data source.
        path : str or list of str
            Path(s) to the NetCDF files. Glob patterns are expanded.
        variables : list of str, optional
            The variables to read. By default, all the variables along the
            point dimension, except the latitude, longitude and time ones.
        flavour : dict, optional
            Names of the ``latitude``, ``longitude`` and ``date`` variables in
            the files. Defaults to ``latitude``, ``longitude`` and ``time``.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(context, *args, **kwargs)

        self.flavour = {
            "latitude": "latitude",
            "longitude": "longitude",
            "date": "time",
        }
        if flavour is not None:
            self.flavour.update(flavour)

        self.paths = sorted(set(_expand(path)))
        if not self.paths:
            raise ValueError(f"No NetCDF files found for {path}")

        self.variables = list(variables) if variables is not None else self._default_variables(self.paths[0])
        self.columns = ["date", "latitude", "longitude"] + self.variables

        self._time_ranges: dict[str, tuple[np.datetime64, np.datetime64] | None] = {}

    def _default_variables(self, path: str) -> list[str]:
        import xarray as xr

        coords = set(self.flavour.values())
        with xr.open_dataset(path) as ds:
            dim = ds[self.flavour["latitude"]].dims
            return [name for name, var in ds.data_vars.items() if name not in coords and var.dims == dim]

    def _time_range(self, path: str) -> tuple[np.datetime64, np.datetime64] | None:
        """Return the (min, max) observation times of a file, or None if it has none."""
        import xarray as xr

        if path not in self._time_ranges:
            with xr.open_dataset(path) as ds:
                times = ds[self.flavour["date"]].values
            times = times[~np.isnat(times)]
            self._time_ranges[path] = (times.min(), times.max()) if len(times) else None

        return self._time_ranges[path]

    def _read(self, path: str, start: np.datetime64, end: np.datetime64) -> pd.DataFrame:
        import xarray as xr

        names = {
            self.flavour["date"]: "date",
            self.flavour["latitude"]: "latitude",
            self.flavour["longitude"]: "longitude",
        }

        with xr.open_dataset(path) as ds:
            frame = pd.DataFrame({new: ds[old].values for old, new in names.items()})
            for name in self.variables:
                frame[name] = ds[name].values

        frame["latitude"] = frame["latitude"].astype(np.float64)
        frame["longitude"] = frame["longitude"].astype(np.float64)

        mask = (frame["date"] >= start) & (frame["date"] <= end)
        return frame.loc[mask, self.columns]

    def execute(self, dates: Any) -> pd.DataFrame:
        # Both ends of the range are inclusive
        start, end = np.datetime64(dates.start_range, "ns"), np.datetime64(dates.end_range, "ns")

        frames = []
        for path in self.paths:
            time_range = self._time_range(path)
            if time_range is None or time_range[1] < start or time_range[0] > end:
                continue
            frames.append(self._read(path, start, end))

        if frames:
            frame = pd.concat(frames, ignore_index=True)
        else:
            frame = pd.DataFrame({name: pd.Series(dtype=np.float64) for name in self.columns})
            frame["date"] = pd.Series(dtype="datetime64[ns]")

        self.context.trace(self.emoji, f"netcdf-tabular: {len(frames)} file(s), {len(frame):,} rows")
        return frame
