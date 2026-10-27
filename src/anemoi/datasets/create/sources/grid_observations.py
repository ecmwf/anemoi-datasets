# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

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

    Parameters
    ----------
    input : dict
        The tabular input (a source or a pipe).
    window : str
        The window of date T, relative to T (anemoi.utils.window.Window).
    frequency : str, optional
        The step that ``time_delta_norm`` divides by; the dataset frequency by default.
    **gridding
        ``anemoi.transform.obs.Gridding``: grid, group_by, select, qc, variables, maps.
    """

    emoji = "🗺️"  # For tracing

    def __init__(self, context, input: dict, window: str, frequency: str | None = None, **gridding):
        from anemoi.transform.obs import Gridding

        super().__init__(context)
        self.input = context.create_source(input, "data_sources", str(id(self)))
        self.window = Window(window)
        self.window_string = window
        if frequency is None:
            frequency = context.recipe.dates.frequency
        self.frequency = frequency_to_timedelta(frequency)
        self.gridding = Gridding(**gridding)

    def execute_valid_dates(self, argument) -> ekd.FieldList:
        from anemoi.transform.obs import grid_windows
        from anemoi.transform.obs.grid import grid_points

        from anemoi.datasets.create.arguments import TimeSpan

        steps = pd.DatetimeIndex(sorted(argument.dates))
        span = TimeSpan.from_window(self.window, list(steps.to_pydatetime()), self.gridding.input_columns())
        rows = self.input(self.context, span)
        if not isinstance(rows, pd.DataFrame):
            raise TypeError(f"grid_observations: the input must give a DataFrame, not {type(rows).__name__}")
        out = grid_windows(self.gridding, rows, steps, self.window, self.frequency)
        self.record_diagnostics(out, steps)
        self.context.trace(
            self.emoji, f"{len(rows)} rows → {len(out.picked)} cells over {len(steps)} windows ({self.window_string})"
        )
        lat, lon = grid_points(self.gridding.grid)
        fields = [
            {
                "param": name,
                "values": array[i],
                "latitudes": lat,
                "longitudes": lon,
                "valid_datetime": time.to_pydatetime(),
            }
            for i, time in enumerate(steps)
            for name, array in out.arrays.items()
        ]
        assert all(f["values"].dtype == np.float32 for f in fields)
        return ekd.from_source("list-of-dicts", fields)

    def record_diagnostics(self, out, steps: pd.DatetimeIndex) -> None:
        """What `anemoi-datasets monitor` plots: per day and group, the selected values'
        stats and the QC counts of the rows read; per cell, rows read and rejected by QC."""
        if not hasattr(self.context, "add_diagnostics"):
            return
        observed = self.gridding.variables.observed()
        self.context.add_diagnostics("obs_stats", out.stats(observed, self.gridding.group_by))
        self.context.add_diagnostics("obs_qc", out.qc.assign(day=steps[0].floor("D")))
        self.context.add_diagnostics("obs_qc_cells", out.sums["qc"].astype(np.int32))  # rows, rejected
