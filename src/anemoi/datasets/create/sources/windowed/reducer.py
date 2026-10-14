# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Reduce one window of interval-valued source data to one field.

A window is cut into subwindows (:mod:`.subwindows`). Each is rebuilt from archived
fields by a **signed sum** -- one field when the archive holds it outright, ``+a(0,12)
-a(0,6)`` when the archive accumulates from the start of the run. Once a subwindow is
complete its value is reduced into the window value by the operation (sum, max, min,
mean), and released.
"""

import datetime
import logging

import numpy as np
from anemoi.transform.fields import Field
from anemoi.utils.dates import frequency_to_string
from numpy.typing import NDArray

from anemoi.datasets.create.intervals import SignedInterval

from .operations import Operation
from .operations import stamp
from .subwindows import Subwindow

LOG = logging.getLogger(__name__)


def _matches(wanted: SignedInterval, interval: SignedInterval) -> bool:
    if wanted.min == interval.min and wanted.max == interval.max and wanted.base == interval.base:
        return True
    return wanted.start == interval.start and wanted.end == interval.end and wanted.base is None


class Reducer:
    """One ``(valid_date, basetime, variable)`` window.

    Parameters
    ----------
    valid_date : datetime.datetime
        The end of the window, which the output is stamped with.
    period : datetime.timedelta
        The length of the window.
    key : tuple
        The grouping key identifying the variable.
    subwindows : list of Subwindow
        The pieces the window is cut into; every one is required.
    operation : Operation
        What reduces the subwindow values.
    basetime : datetime.datetime, optional
        The model-run time to stamp the output with, for a trajectory row.
    """

    def __init__(
        self,
        valid_date: datetime.datetime,
        period: datetime.timedelta,
        key: tuple,
        subwindows: list[Subwindow],
        operation: Operation,
        basetime: datetime.datetime | None = None,
    ) -> None:
        self.valid_date = valid_date
        self.period = period
        self.key = key
        self.operation = operation
        self.basetime = basetime

        self.subwindows = subwindows
        self.todo = [list(s.contributions) for s in subwindows]
        self._partial: list[NDArray | None] = [None] * len(subwindows)
        self._reduced: NDArray | None = None
        self._total_weight = 0.0
        self.outstanding = len(subwindows)
        #: How many fields some subwindow took. "No field arrived" and "no subwindow
        #: completed" are different: a differenced subwindow takes two fields first.
        self.fields_used = 0
        self.locked = False

    def is_complete(self) -> bool:
        return self.outstanding == 0

    def compute(self, values: NDArray, interval: SignedInterval) -> bool:
        """Offer a field to every subwindow of this window; True if one took it.

        One field can serve several subwindows: on a cumulative archive ``a(0,7)`` is
        ``+`` in ``[6,7]`` and ``-`` in ``[7,8]``. *values* is shared with every other
        window, so it is never modified in place.
        """
        used = False
        for i, todo in enumerate(self.todo):
            matching = [c for c in todo if _matches(c, interval)]
            if not matching:
                continue
            used = True
            for contribution in matching:
                local = contribution.sign * values  # a new array, never the shared one
                if self._partial[i] is None:
                    self._partial[i] = local
                else:
                    self._partial[i] += local
                todo.remove(contribution)

            if not todo:
                subwindow = self.subwindows[i]
                weight = (subwindow.end - subwindow.start).total_seconds()
                self._reduced = self.operation.reduce(self._reduced, self._partial[i], weight)
                self._partial[i] = None
                self._total_weight += weight
                self.outstanding -= 1

        if used:
            self.fields_used += 1
        return used

    def as_field(self, template: Field) -> Field:
        """Build the reduced field, once every subwindow is in."""
        assert self.is_complete(), self.__repr__(verbose=True)
        assert not self.locked  # prevent building the field twice

        values = self.operation.finalize(self._reduced, self._total_weight)

        # negative values may be an anomaly (e.g precipitation), but this is user's choice
        if self.operation.name == "sum" and ("param", "tp") in self.key and np.any(values < 0):
            LOG.warning(
                f"Negative values when computing accumulation for {self}): "
                f"min={np.nanmin(values)} max={np.nanmax(values)}"
            )

        self.locked = True
        return stamp(values, template, self.operation, self.valid_date, self.period, self.basetime)

    def __repr__(self, verbose: bool = False) -> str:
        key = ", ".join(f"{k}={v}" for k, v in self.key)
        period = frequency_to_string(self.period)
        run = f", basetime={self.basetime}" if self.basetime is not None else ""
        default = (
            f"{type(self).__name__}(valid_date={self.valid_date}{run}, {period}, "
            f"{self.operation.name}, key={{ {key} }})"
        )
        if verbose:
            extra = ["    (locked)"] if self.locked else []
            for subwindow, todo in zip(self.subwindows, self.todo):
                extra.append(f"    {subwindow}: {'done' if not todo else 'waiting for'}")
                extra.extend(f"        {'+' if c.sign > 0 else '-'}{c}" for c in todo)
            default += "\n" + "\n".join(extra)
        return default


class Logs(list):
    def __init__(self, *args, reducers, source, source_object, field_to_interval, **kwargs):
        super().__init__(*args, **kwargs)
        self.reducers = reducers
        self.source = source
        self.source_object = source_object
        self.field_to_interval = field_to_interval

    def raise_error(self, msg, field=None, field_interval=None) -> str:
        INTERVAL_COLOR = "\033[93m"
        FIELD_COLOR = "\033[92m"
        KEY_COLOR = "\033[95m"
        RESET_COLOR = "\033[0m"

        res = [""]
        res.append(f"❌ {msg}")
        res.append(f"💬 Patches applied: {self.field_to_interval.patches}")
        res.append("💬 Current field:")
        res.append(f" {FIELD_COLOR}{field}{RESET_COLOR}")
        res.append(f" {INTERVAL_COLOR}{field_interval}{RESET_COLOR}")
        if self.reducers:
            res.append(f"💬 Existing reducers ({len(self.reducers)}) :")
            for a in self.reducers.values():
                res.append(f"  {a.__repr__(verbose=True)}")
        res.append(f"💬 Received fields ({len(self)}):")
        for log in self:
            res.append(f"  {KEY_COLOR}{log[0]}{RESET_COLOR} {INTERVAL_COLOR}{log[2]}{RESET_COLOR}")
            res.append(f"       {KEY_COLOR}{log[1]}{RESET_COLOR}")
            if log[3]:
                res.append("   used for " + ", ".join(str(d) for d in log[3]))
            else:
                res.append("   used for nothing")

        LOG.error("\n".join(res))
        res = ["More details below:"]

        res.append(f"💬 Fields returned to be accumulated ({len(self.source_object)}):")
        for field in self.source_object:
            res.append(
                f"  {field}, startStep={field.metadata('startStep')}, endStep={field.metadata('endStep')} mean={np.nanmean(field.values, axis=0)}"
            )

        LOG.error("\n".join(res))
        res = ["Even more details below:"]

        if "mars" in self.source:
            res.append("💬 Example of code fetching some available fields and inspect them:")
            res.append("# --------------------------------------------------")
            code = []
            code.append("from earthkit.data import from_source")
            code.append("import numpy as np")
            code.append('ds = from_source("mars", **{')
            for k, v in self.source["mars"].items():
                code.append(f"    {k!r}: {v!r},")
            code.append(f'    "date": {field.metadata("date")!r},')
            code.append(f'    "time": {field.metadata("time")!r}, # "ALL"')
            code.append(f'    "step": "ALL", # {field.metadata("step")!r},')
            code.append("})")
            code.append('print(f"Got {len(ds)} fields:")')
            code.append("prev_m = None")
            code.append("for field in ds[:50]: # limit to first 50 for brevity")
            code.append(
                '    print(f"{field} startStep={field.metadata("startStep")}, endStep={field.metadata("endStep")} mean={np.nanmean(field.values)}")'
            )
            res.append("# --------------------------------------------------")
            code.append("")
            res += code

            # now execute the code to show actual field values
            LOG.error("\n".join(res))

        raise ValueError(msg)
