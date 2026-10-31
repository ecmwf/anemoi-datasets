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
import re

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.window import Window

from ..source import Source
from . import source_registry

LOG = logging.getLogger(__name__)


@source_registry.register("grid_observations")
class GridObservationsSource(Source):
    """Observations gridded per time window: each date T gets, per grid cell, one observation
    (or one overpass mean) of the rows in T's window, plus derived geometry and forcings.

    The inner ``input`` is any tabular source or pipe (``parquet``, ``odb``, ... and tabular
    filters). It is called once per group with a ``TimeSpan`` covering every window of the
    group, so windows that reach into the previous day need no extra configuration. The
    gridding itself is ``anemoi.transform.obs`` (pure, shared with real-time processing).

    ``inputs`` instead combines several: each item is an ``input`` plus how its rows join
    the dataset (``anemoi.transform.obs.Input``: dates, groups, channels, group_offset,
    scan). Items with the same ``input`` read it once; an input whose dates miss the group
    is not read.

    Examples
    --------
    .. code-block:: yaml

      grid_observations:
        window: "(-350m,+10m]"    # T - 5h50 < date <= T + 10min
        grid: o96
        select: {candidates: latest_pass, reduce: nearest}
        qc: {flags: {gqis_flag_qual: [0]}}
        variables: {channels: [bt_16], columns: [satid], forcings: [cos_sza]}
        input:
          parquet: {path: ..., flavour: {date: datetime, latitude: lat, longitude: lon}}

    .. code-block:: yaml

      grid_observations:
        ...
        inputs:
          - input: {parquet: {path: .../msu_noaa_cdr/...}}
            dates: [1979-01-01, 2006-11-01]
            channels: {bt_1: bt_3, bt_2: bt_5}
            group_offset: 10000
          - input: {parquet: {path: .../amsua_od_ai/...}}
            dates: [1998-08-01, null]

    Parameters
    ----------
    context : Any
        The build context.
    window : str
        The window of date T, relative to T (anemoi.utils.window.Window).
    input : dict, optional
        The tabular input (a source or a pipe).
    inputs : list[dict], optional
        Several inputs instead, each ``{input: ..., **anemoi.transform.obs.Input}``.
    frequency : str, optional
        The step that ``time_delta_norm`` divides by; the dataset frequency by default.
    **gridding
        ``anemoi.transform.obs.Gridding``: grid, group_by, select, qc, variables, maps.
    """

    emoji = "🗺️"  # For tracing

    def __init__(
        self,
        context,
        window: str,
        input: dict | None = None,
        inputs: list[dict] | None = None,
        frequency: str | None = None,
        **gridding,
    ):
        from anemoi.transform.obs import Gridding
        from anemoi.transform.obs import Input

        super().__init__(context)
        if (input is None) == (inputs is None):
            raise ValueError("grid_observations: set one of input and inputs")
        self.window = Window(window)
        self.window_string = window
        if frequency is None:
            frequency = context.recipe.dates.frequency
        self.frequency = frequency_to_timedelta(frequency)
        self.gridding = Gridding(**gridding)
        self.sources = {}  # one source per distinct input config
        self.inputs = []  # (Input, key of its source)
        for item in inputs if inputs is not None else [{"input": input}]:
            item = dict(item)
            config = item.pop("input")
            key = json.dumps(config, sort_keys=True, default=str)
            if key not in self.sources:
                self.sources[key] = context.create_source(config, "data_sources", f"{id(self)}-{len(self.sources)}")
            self.inputs.append((Input(**item), key))
            self.inputs[-1][0].check(self.gridding)

    def execute_valid_dates(self, argument) -> ekd.FieldList:
        from anemoi.transform.obs import grid_windows
        from anemoi.transform.obs.grid import grid_points

        steps = pd.DatetimeIndex(sorted(argument.dates))
        rows = self.read(steps)
        out = grid_windows(self.gridding, rows, steps, self.window, self.frequency)
        self.record_diagnostics(out, steps)
        self.context.trace(
            self.emoji, f"{len(rows)} rows → {len(out.picked)} cells over {len(steps)} windows ({self.window_string})"
        )
        lat, lon = grid_points(self.gridding.grid)
        grid = self.gridding.grid
        resolution = grid.upper() if re.fullmatch(r"[onON]\d+", grid) else grid  # as GRIB: O96, N320
        fields = [
            {
                "param": name,
                "values": array[i],
                "latitudes": lat,
                "longitudes": lon,
                "valid_datetime": time.to_pydatetime(),
                "resolution": resolution,
            }
            for i, time in enumerate(steps)
            for name, array in out.arrays.items()
        ]
        assert all(f["values"].dtype == np.float32 for f in fields)
        return ekd.from_source("list-of-dicts", fields)

    def read(self, steps: pd.DatetimeIndex) -> pd.DataFrame:
        """The rows of every input for the windows of `steps`, combined (Input.prepare)."""
        from anemoi.transform.obs import combine

        from anemoi.datasets.create.arguments import TimeSpan

        dates = list(steps.to_pydatetime())
        span = TimeSpan.from_window(self.window, dates)
        wanted = [(inp, key) for inp, key in self.inputs if inp.overlaps(span.start, span.end)]
        parts = []
        for key in dict.fromkeys(k for _, k in wanted):
            using = [inp for inp, k in wanted if k == key]
            columns = list(dict.fromkeys(c for inp in using for c in inp.columns(self.gridding)))
            rows = self.sources[key](self.context, TimeSpan.from_window(self.window, dates, columns))
            if not isinstance(rows, pd.DataFrame):
                raise TypeError(f"grid_observations: an input must give a DataFrame, not {type(rows).__name__}")
            parts += [(inp, rows) for inp in using]
        return combine(self.gridding, parts)

    def record_diagnostics(self, out, steps: pd.DatetimeIndex) -> None:
        """What `anemoi-datasets monitor` plots: per day and group, the selected values'
        stats and the QC counts of the rows read; per cell, rows read and rejected by QC.
        """
        if not hasattr(self.context, "add_diagnostics"):
            return
        v = self.gridding.variables
        observed = [v.final(n) for n in v.observed()]
        self.context.add_diagnostics("obs_stats", out.stats(observed, self.gridding.group_by))
        self.context.add_diagnostics("obs_qc", out.qc.assign(day=steps[0].floor("D")))
        self.context.add_diagnostics("obs_qc_cells", out.sums["qc"].astype(np.int32))  # rows, rejected
