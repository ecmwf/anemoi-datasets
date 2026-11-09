# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from typing import Any

import numpy as np
from anemoi.utils.dates import frequency_to_string
from numpy.typing import NDArray

from anemoi.datasets.create.intervals import SignedInterval

from .writers import write_accumulated_field_with_valid_time
from .writers import write_accumulated_forecast_field

LOG = logging.getLogger(__name__)


class Accumulator:
    values: NDArray | None = None
    locked: bool = False

    def __init__(
        self,
        valid_date: datetime.datetime,
        period: datetime.timedelta,
        key: dict[str, Any],
        coverage,
        basetime: datetime.datetime | None = None,
    ):
        # The accumulator only accumulates fields and does not know about the rest
        # Accumulator object for a given param/member/valid_date

        self.valid_date = valid_date
        self.period = period
        self.key = key
        self.basetime = basetime

        self.coverage = coverage

        self.todo = [v for v in coverage]
        self.done = []

        self.values = None  # will hold accumulated values array

    def is_complete(self, **kwargs) -> bool:
        """Check whether the accumulation is complete (all intervals have been processed)"""
        return not self.todo

    def compute(self, values: NDArray, interval: SignedInterval) -> None:
        """Perform accumulation with the values array on this interval and record the operation.
        Note: values have been extracted from field before the call to `compute`,
        so values are read from field only once.

        Parameters:
        ----------
        field: Any
            An earthkit-data-like field
        values: NDArray
            Values from the field, will be added to the held values array

        Return
        ------
        None
        """

        def match_interval(interval: SignedInterval, lst: list[SignedInterval]) -> bool:
            for i in lst:
                if i.min == interval.min and i.max == interval.max and i.base == interval.base:
                    return i
                if i.start == interval.start and i.end == interval.end and i.base is None:
                    return i
            return None

        matching = match_interval(interval, self.todo)

        if not matching:
            # interval not needed for this accumulator
            # this happens when multiple accumulators have the same key but different valid_date
            return False

        def raise_error(msg):
            LOG.error(f"Accumulator {self.__repr__(verbose=True)} state:")
            LOG.error(f"Received interval: {interval}")
            LOG.error(f"Matching interval: {matching}")
            raise ValueError(msg)

        if matching in self.done:
            # this should not happen normally
            raise_error(f"SignedInterval {matching} already done for accumulator")

        if self.locked:
            raise_error(f"Accumulator already used, cannot process interval {interval}")

        assert isinstance(values, np.ndarray), type(values)

        # actual accumulation computation
        # negative accumulation if interval is reversed
        # copy is mandatory since value is shared between accumulators
        local_values = matching.sign * values.copy()
        if self.values is None:
            self.values = local_values
        else:
            self.values += local_values

        self.todo.remove(matching)
        self.done.append(matching)
        return True

    def write_to_output(self, output, template) -> None:
        assert self.is_complete(), (self.todo, self.done, self)
        assert not self.locked  # prevent double writing

        # negative values may be an anomaly (e.g precipitation), but this is user's choice
        for k, v in self.key:
            if k == "param" and v == "tp":
                if np.any(self.values < 0):
                    LOG.warning(
                        f"Negative values when computing accumutation for {self}): min={np.nanmin(self.values)} max={np.nanmax(self.values)}"
                    )
        if self.basetime is not None:
            write_accumulated_forecast_field(
                template=template,
                values=self.values,
                basetime=self.basetime,
                valid_date=self.valid_date,
                period=self.period,
                output=output,
            )
        else:
            write_accumulated_field_with_valid_time(
                template=template,
                values=self.values,
                valid_date=self.valid_date,
                period=self.period,
                output=output,
            )
        # lock the accumulator to prevent further use
        self.locked = True

    def __repr__(self, verbose: bool = False) -> str:
        key = ", ".join(f"{k}={v}" for k, v in self.key)
        period = frequency_to_string(self.period)
        default = f"{self.__class__.__name__}(valid_date={self.valid_date}, {period}, key={{ {key} }})"
        if verbose:
            extra = []
            if self.locked:
                extra.append("(locked)")
            for i in self.done:
                extra.append(f"    done: {i}")
            for i in self.todo:
                extra.append(f"    todo: {i}")
            default += "\n" + "\n".join(extra)
        return default


class Logs(list):
    def __init__(self, *args, accumulators, source, source_object, field_to_interval, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulators = accumulators
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
        if self.accumulators:
            res.append(f"💬 Existing accumulators ({len(self.accumulators)}) :")
            for a in self.accumulators.values():
                res.append(f"  {a.__repr__(verbose=True)}")
        res.append(f"💬 Received fields ({len(self)}):")
        for log in self:
            res.append(f"  {KEY_COLOR}{log[0]}{RESET_COLOR} {INTERVAL_COLOR}{log[2]}{RESET_COLOR}")
            res.append(f"       {KEY_COLOR}{log[1]}{RESET_COLOR}")
            for d, acc_repr in zip(log[3], log[4]):
                res.append(f"   used for date {d}: {acc_repr}")

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
