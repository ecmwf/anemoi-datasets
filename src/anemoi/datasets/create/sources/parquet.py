# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
import os

import pandas as pd
from anemoi.utils.window import Window

from ..source import Source
from . import source_registry

LOG = logging.getLogger(__name__)


@source_registry.register("parquet")
class ParquetSource(Source):
    """Observation rows from per-day parquet files, as a DataFrame.

    Only the day files a time range touches are opened, and the time filter is pushed down
    to pyarrow, so files with one row group per hour skip the hours outside the range.

    Examples
    --------
    .. code-block:: yaml

      parquet:
        path: /ec/project/dop/base_data/iasi_eumetsat_l1c/year=%Y/month=%m/%Y-%m-%d.parquet
        flavour: {date: datetime, latitude: lat, longitude: lon}
        columns: [satid, bt_16]

    Parameters
    ----------
    path : str
        A strftime template of one day's file.
    columns : list[str], optional
        Columns to read (anemoi names, see ``flavour``), on top of those a calling source
        asks for (``TimeSpan.columns``). Every column if neither gives any.
    flavour : dict, optional
        File column of each of ``date``, ``latitude`` and ``longitude``.
    window : str, optional
        Needed when called with dates rather than a time range: the rows of
        ``(first date + window start, last date + window end]``, e.g. "(-3h,+3h]".
    missing_files : str
        "error" (default) or "skip": a day without a file has no rows.
    """

    emoji = "🧱"  # For tracing

    def __init__(
        self,
        context,
        path: str,
        columns: list[str] | None = None,
        flavour: dict | None = None,
        window: str | None = None,
        missing_files: str = "error",
        **kwargs,
    ):
        super().__init__(context, **kwargs)
        if missing_files not in ("error", "skip"):
            raise ValueError(f"parquet: missing_files must be 'error' or 'skip', not {missing_files!r}")
        self.path = path
        self.columns = list(columns or [])
        self.flavour = {"date": "date", "latitude": "latitude", "longitude": "longitude"} | (flavour or {})
        self.window = Window(window) if window else None
        self.missing_files = missing_files

    def execute_time_span(self, argument):
        return self.read(argument.start, argument.end, argument.closed, argument.columns)

    def execute_valid_dates(self, argument):
        from anemoi.datasets.create.arguments import TimeSpan

        if self.window is None:
            raise ValueError("parquet: called with dates, set `window` to say which rows each date reads")
        return self.execute_time_span(TimeSpan.from_window(self.window, argument.dates))

    def files(self, start: datetime.datetime, end: datetime.datetime) -> list[str]:
        days = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"))
        files = [d.strftime(self.path) for d in days]
        missing = [f for f in files if not os.path.exists(f)]
        if missing and self.missing_files == "error":
            raise FileNotFoundError(f"parquet: {len(missing)} day files missing, e.g. {missing[0]}")
        return [f for f in files if f not in missing]

    def read(self, start, end, closed: str = "right", columns: list[str] | None = None) -> pd.DataFrame:
        """Rows from start to end, ends included as `closed` says (pandas.Interval)."""
        import pyarrow.dataset as ds

        wanted = list(dict.fromkeys([*(columns or []), *self.columns])) or None
        if wanted is not None:
            wanted = list(dict.fromkeys(["date", "latitude", "longitude", *wanted]))
        to_file = dict(self.flavour)
        file_columns = None if wanted is None else [to_file.get(c, c) for c in wanted]

        start, end = pd.Timestamp(start), pd.Timestamp(end)
        files = self.files(start, end)
        self.context.trace(self.emoji, f"parquet {start}..{end} ({closed}): {len(files)} files")
        if not files:
            return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "date" else float) for c in wanted or []})

        t = ds.field(to_file["date"])
        after = t >= start if closed in ("left", "both") else t > start
        before = t <= end if closed in ("right", "both") else t < end
        table = ds.dataset(files, format="parquet").to_table(columns=file_columns, filter=after & before)
        frame = table.to_pandas()
        return frame.rename(columns={v: k for k, v in to_file.items() if v != k})
