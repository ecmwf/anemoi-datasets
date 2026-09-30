# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Per-window state: fill each part of the window, then reduce the parts.

One :class:`Reducer` per ``(valid_date, basetime, variable)``. It holds one
:class:`~.states.State` per part -- a subwindow or a sample -- and works at two levels:

- a **state** gathers the fields its part needs and makes one value; for a subwindow
  that is a signed sum, which is where differencing happens;
- the **operation** reduces each completed value into the window value, and the state's
  array is released.

Peak memory is therefore bounded by the parts still incomplete, not by the window.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import numpy as np
from anemoi.transform.fields import Field
from anemoi.utils.dates import frequency_to_string

from .operations import Operation
from .operations import operation_factory
from .states import NON_ADDITIVE
from .states import State

LOG = logging.getLogger(__name__)


class Reducer:
    """Reduce one window to one field.

    Parameters
    ----------
    valid_date : datetime.datetime
        The validity time the output is stamped with (the end of the window).
    period : datetime.timedelta
        The length of the window.
    key : tuple
        The grouping key: the metadata identifying the variable.
    states : list of State
        The parts the window is made of; all of them are required.
    operation : Operation or str, optional
        What reduces the state values. Defaults to a sum.
    basetime : datetime.datetime, optional
        The model-run time to stamp the output with, for a trajectory row. ``None``
        stamps the start of the window instead, so the whole step is the window.
    """

    def __init__(
        self,
        valid_date: datetime.datetime,
        period: datetime.timedelta,
        key: tuple,
        states: list[State],
        operation: Operation | str | None = None,
        basetime: datetime.datetime | None = None,
    ) -> None:
        self.valid_date = valid_date
        self.period = period
        self.key = key
        self.basetime = basetime
        self.operation = operation_factory(operation)

        self.states = list(states)
        self._reduced: np.ndarray | None = None
        self._done = 0
        #: How many arriving fields some state took. Distinct from the reduced value
        #: being set, which needs a whole *part* to complete: a subwindow rebuilt by
        #: differencing takes several fields before it yields anything. "Nothing
        #: arrived" and "nothing finished" are different failures.
        self.fields_used = 0
        self.total_weight = 0.0
        self.locked = False

    @property
    def values(self) -> np.ndarray | None:
        """The window value so far; ``None`` until the first state completes."""
        return self._reduced

    def is_complete(self) -> bool:
        """Whether every state has been filled and reduced in."""
        return self._done == len(self.states)

    def compute(self, values: np.ndarray, identity: Any, statistic: str | None = None) -> bool:
        """Offer a field to this window's states.

        A field can fill more than one state: with a cumulative archive neighbouring
        subwindows share an endpoint, so ``a(0,7)`` is the positive contribution of
        ``[6,7]`` and the negative one of ``[7,8]``.

        Parameters
        ----------
        values : numpy.ndarray
            Values read off the field. Shared with every other window this field feeds,
            so states copy before taking ownership.
        identity : Any
            What the field is, in whatever terms the states are expressed.
        statistic : str, optional
            What statistic the field carries, when known. Used to reject a
            non-additive field being handed to a state that would difference it.

        Returns
        -------
        bool
            True if some state needed this field.
        """
        # What the archive stores has to be something this reduction can be applied to,
        # whatever the partition looks like. Checked before the states, because it is a
        # property of the source data and the recipe rather than of any one part.
        if statistic is not None and not self.operation.reduces_archived(statistic):
            raise ValueError(f"{self!r}: {self.operation.why_not_archived(statistic)}")

        # No guard on `locked` here: once every state is complete none of them accepts
        # anything, so a repeat simply reports "not needed". Overlapping windows make
        # repeats ordinary -- the same field is offered once per window that wants it.
        used = False
        for state in self.states:
            if state.is_complete:
                continue

            # The recipe cannot tell us this -- `accumulate` declares no statistic --
            # but the field can. Subtracting two maxima does not give the maximum over
            # the difference of their intervals; it gives nothing meaningful. The state
            # says whether it differences and whether it wants this field; the reducer
            # does not need to know what kind of state it is holding.
            if statistic in NON_ADDITIVE and state.differences and state.wants(identity):
                raise ValueError(f"{self!r}: {state.why_not(statistic)}")

            if state.accept(values, identity):
                used = True
                if state.is_complete:
                    self._reduced = self.operation.reduce(self._reduced, state.release(), state.weight)
                    self.total_weight += state.weight
                    self._done += 1
        if used:
            self.fields_used += 1
        return used

    def as_field(self, template: Field) -> Field:
        """Build the reduced field, once every state is in.

        For a gridded window (no basetime) the base time is the start of the window and
        the step reaches the validity time, so the whole step is the window. For a
        trajectory row the base time is the model-run basetime, so trajectory loaders
        recover ``(basetime, step)``. Either way the processing component records which
        reduction produced it, over ``period``.

        Parameters
        ----------
        template : Field
            Field providing every other component (parameter, geography, ...).

        Returns
        -------
        Field
            The reduced field.
        """
        assert self.is_complete(), (self._done, len(self.states), self)
        assert not self.locked  # prevent building the field twice

        values = self.operation.finalize(self._reduced, self.total_weight)

        # Negative values may be an anomaly (e.g. precipitation), but this is the user's
        # choice. Only meaningful for a sum: an extremum of non-negative fields cannot
        # go negative.
        if self.operation.name == "sum" and any(k == "param" and v == "tp" for k, v in self.key):
            if np.any(values < 0):
                LOG.warning(
                    f"Negative values when computing accumulation for {self}): "
                    f"min={np.nanmin(values)} max={np.nanmax(values)}"
                )

        basetime = self.basetime if self.basetime is not None else self.valid_date - self.period
        field = Field.from_numpy(
            values,
            template=template,
            **{
                "time.base_datetime": basetime,
                "time.step": self.valid_date - basetime,
                "proc.time_method": self.operation.time_method,
                "proc.time_value": self.period,
            },
        )
        self.locked = True
        return field

    def __repr__(self, verbose: bool = False) -> str:
        key = ", ".join(f"{k}={v}" for k, v in self.key)
        period = frequency_to_string(self.period)
        run = f", basetime={self.basetime}" if self.basetime is not None else ""
        operation = "" if self.operation.name == "sum" else f", {self.operation.name}"
        default = f"{type(self).__name__}(valid_date={self.valid_date}{run}, {period}{operation}, key={{ {key} }})"
        if verbose:
            extra = []
            if self.locked:
                extra.append("(locked)")
            extra.append(f"    reduced {self._done} of {len(self.states)}, from {self.fields_used} field(s):")
            for state in self.states:
                extra.append(f"    {state}")
            default += "\n" + "\n".join(extra)
        return default


def describe(reducers: dict, limit: int = 20) -> str:
    """Render reducers for an error message, least complete first."""
    ordered = sorted(reducers.values(), key=lambda r: (r._done - len(r.states), r.valid_date))
    lines = [f"  {r.__repr__(verbose=True)}" for r in ordered[:limit]]
    if len(ordered) > limit:
        lines.append(f"  ... and {len(ordered) - limit} more")
    return "\n".join(lines)


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
