# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The loop for interval-valued source data: each field *spans* an interval.

Used by ``accumulate`` (with ``sum``) and by ``maximum`` / ``minimum`` / ``average``
when ``from:`` describes archived intervals rather than instants. A window is cut into
subwindows by the covering, every field is offered to every window, and a window is
complete when every subwindow has been rebuilt and reduced.
"""

import datetime
import logging
from typing import Any

from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.intervals import SignedInterval

from .check import Pieces
from .check import check
from .check import field_statistic
from .common import group_key
from .operations import Operation
from .reducer import Logs
from .reducer import Reducer
from .subwindows import Subwindow
from .subwindows import contributions_of
from .subwindows import validate_partition

LOG = logging.getLogger(__name__)

#: A build target: ``(valid_date, basetime)``, with *basetime* ``None`` outside a
#: trajectory layout.
Target = tuple


def cut(
    covering: Any,
    start: datetime.datetime,
    end: datetime.datetime,
    basetime: datetime.datetime | None = None,
    over: datetime.timedelta | None = None,
) -> list[Subwindow]:
    """The subwindows of the window ``[start, end]``.

    Without *over* they are whatever the covering finds, so their length is the
    archive's. With *over* the window is cut into ``over``-long slices, each covered on
    its own and rebuilt by one signed sum -- which is what turns an accumulating
    archive into hourly totals for "the wettest hour". Slices are fixed, not rolling.

    *basetime* is passed to the covering only when given: it is the run the trajectory
    layout imposes, which only a ``ForecastCovering`` takes.
    """

    def cover(a, b):
        return list(covering.partition(a, b, basetime=basetime) if basetime else covering.partition(a, b))

    if over is None:
        return cover(start, end)

    subwindows = []
    at = start
    while at < end:
        found = cover(at, at + over)
        if len(found) == 1:
            subwindows.append(found[0])
        else:
            interval = SignedInterval(start=at, end=at + over)
            subwindows.append(Subwindow(interval=interval, contributions=tuple(contributions_of(found))))
        at += over
    validate_partition(subwindows, start, end)
    return subwindows


def argument(targets: list[Target], windows: dict, period: datetime.timedelta, forecast: bool) -> Any:
    """What to ask the subsource for: every archived interval, once.

    Windows overlap whenever ``period`` exceeds the output frequency, so the same
    interval is wanted by several of them; ``+a(0,7)`` and ``-a(0,7)`` are also one
    retrieval. Matching remaps each field onto every window that needs it.
    """
    seen: dict = {}
    for interval in contributions_of(s for t in targets for s in windows[t]):
        seen.setdefault((interval.min, interval.max, interval.base), interval)
    intervals = list(seen.values())

    if forecast:
        return ForecastIntervals(items=[(vt, bt, period) for vt, bt in targets], intervals=intervals)
    return Intervals(dates=sorted({vt for vt, _ in targets}), intervals=intervals)


def reduce_ranges(
    input_fields: Any,
    *,
    targets: list[Target],
    windows: dict,
    period: datetime.timedelta,
    operation: Operation,
    group_by: dict,
    field_to_interval: Any,
    source: dict,
    stamp_basetime: bool,
    declared: str | None = None,
) -> tuple[dict, list]:
    """Reduce every window to one field per variable.

    Parameters
    ----------
    input_fields
        The subsource's fields.
    targets
        The ``(valid_date, basetime)`` rows to produce.
    windows
        Each target's subwindows, from :func:`cut`.
    period
        The window length.
    operation
        What reduces the subwindow values.
    group_by
        Which metadata identifies a variable.
    field_to_interval
        Maps a field to the interval it spans.
    source
        The subsource config, for the diagnostics.
    stamp_basetime
        Whether the output carries the target's basetime (a trajectory row) rather
        than the start of the window.
    declared
        The statistic ``from:`` says the archive stores, used when a field does not.

    Returns
    -------
    tuple
        ``(reducers, fields)``.
    """
    pieces = Pieces.of_subwindows(windows.values())
    checked: set = set()

    reducers: dict = {}
    fields: list = []
    logs = Logs(reducers=reducers, source=source, source_object=input_fields, field_to_interval=field_to_interval)

    for field in input_fields:
        key = group_key(field, group_by)
        interval = field_to_interval(field)

        if key not in checked:
            statistic = field_statistic(field)
            check(operation.name, declared if statistic == "instant" and declared else statistic, pieces)
            checked.add(key)

        # Runs for every field and is thrown away unless something fails, so it keeps
        # only what cannot be recovered later -- no snapshot of every reducer.
        meta = field.get(collections="metadata.mars")
        used_by: list = []
        logs.append([str(field), " ".join(f"{k}={v}" for k, v in meta.items()), interval, used_by])

        values = field.values
        for target in targets:
            reducer_key = (*target, key)
            if reducer_key not in reducers:
                reducers[reducer_key] = Reducer(
                    target[0],
                    period=period,
                    key=key,
                    subwindows=windows[target],
                    operation=operation,
                    basetime=target[1] if stamp_basetime else None,
                )
            reducer = reducers[reducer_key]
            if reducer.compute(values, interval):
                used_by.append(target)
                if reducer.is_complete():
                    fields.append(reducer.as_field(template=field))

        if not used_by:
            logs.raise_error("Field not used for any window", field=field, field_interval=interval)

    return reducers, fields
