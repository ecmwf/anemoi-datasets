# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The plan for instant-valued source data -- ``from: {frequency: ...}``.

A window is the sample times ``(valid - period, valid]`` on the source cadence, and a
field is identified by its validity time. Run-anchored source data identifies a field
by ``(validity time, basetime)`` as well: the same validity time reached from two runs
is two different fields and must never be reduced together.

Used by ``average`` / ``minimum`` / ``maximum``; it lives here beside
:mod:`.interval_plan` because a plan is shared machinery, not a recipe key's business.
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from typing import Any

from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates

from .plan import Target
from .plan import WindowPlan
from .reducer import Reducer
from .samples import Sample
from .states import SampleState

LOG = logging.getLogger(__name__)


def valid_datetime_of(field: Any) -> datetime.datetime:
    """The validity time of *field*.

    The earthkit time component knows it for every field shape; GRIB-backed fields
    that predate that component are read from ``validityDate``/``validityTime``.
    """
    try:
        return field.time.valid_datetime()
    except AttributeError:
        date_str = str(field.metadata("validityDate")).zfill(8)
        time_str = str(field.metadata("validityTime")).zfill(4)
        return datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")


def base_datetime_of(field: Any) -> datetime.datetime:
    """The model-run base time of *field*.

    The mars ``date``/``time`` keys give it for ordinary forecasts, but for hindcast
    fields ``date`` is the reforecast reference date while the run starts at ``hdate``;
    the field's time component is right in both cases.
    """
    try:
        return field.time.base_datetime()
    except AttributeError:
        date_str = str(field.metadata("date")).zfill(8)
        time_str = str(field.metadata("time")).zfill(4)
        return datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")


class SamplingPlan(WindowPlan):
    """Reduce the instantaneous fields sampled inside each window.

    Parameters
    ----------
    period : datetime.timedelta
        The reduction window.
    frequency : datetime.timedelta
        The cadence of the source data.
    operation : str
        The reduction to perform.
    run_anchored : bool
        Whether the source data is the run the trajectory layout imposes.
    name : str
        The recipe key, for error messages.
    """

    def __init__(self, period, frequency, operation: str, run_anchored: bool, name: str) -> None:
        self.period = period
        self.frequency = frequency
        self.operation = operation
        self.run_anchored = run_anchored
        self.name = name
        self._needed_by: dict = {}

    def _sample_id(self, valid_datetime, basetime) -> tuple:
        """The identity a sample is matched on.

        Base-less source data is identified by validity time alone; run-anchored data
        by ``(validity time, basetime)``.
        """
        return (valid_datetime, basetime if self.run_anchored else None)

    def parts_for(self, targets: list[Target]) -> dict[Target, list]:
        from .description import window_samples

        parts = {t: [Sample(s) for s in window_samples(t[0], self.period, self.frequency)] for t in targets}

        # Windows overlap whenever `period` exceeds the output frequency, so one field
        # commonly feeds several of them.
        self._needed_by = defaultdict(list)
        for target in targets:
            for sample in parts[target]:
                self._needed_by[self._sample_id(sample.valid_datetime, target[1])].append(target)

        LOG.debug(
            "%s: %d target(s) x %s / %s -> %d source sample(s)",
            self.name,
            len(targets),
            frequency_to_string(self.period),
            frequency_to_string(self.frequency),
            len(self._needed_by),
        )
        return parts

    def argument(self, targets: list[Target], parts: dict[Target, list]) -> Any:
        if self.run_anchored:
            # Each sample belongs to the run of its own row.
            return ForecastDates(
                sorted({(sample.valid_datetime, t[1]) for t in targets for sample in parts[t]})
            )
        return ValidDates(sorted({sample.valid_datetime for t in targets for sample in parts[t]}))

    def identify(self, field: Any) -> Any:
        basetime = base_datetime_of(field) if self.run_anchored else None
        return self._sample_id(valid_datetime_of(field), basetime)

    def candidates(self, identity: Any, targets: list[Target]) -> list[Target]:
        return self._needed_by.get(identity, ())

    def new_reducer(self, target: Target, key: tuple, parts: list) -> Reducer:
        return Reducer(
            target[0],
            period=self.period,
            key=key,
            states=[SampleState(sample) for sample in parts],
            operation=self.operation,
            basetime=target[1],
        )

    def offer(self, reducer: Reducer, values: Any, identity: Any, info: Any) -> bool:
        # No `info`: an instantaneous field is taken as it is, so there is no
        # differencing for a non-additive statistic to be wrong about.
        return reducer.compute(values, identity)

    def unused_field(self, field: Any, identity: Any) -> None:
        if self._needed_by.get(identity):
            # Some window wanted this sample, so every one of them already has it.
            raise ValueError(
                f"{self.name}: sample {identity[0]} was already reduced into every window "
                f"that needs it; the source returned {field} twice"
            )
        run = f" of the run based at {identity[1]}" if self.run_anchored else ""
        raise ValueError(
            f"{self.name}: field {field} (valid {identity[0]}{run}) is not part of any "
            f"window. The windows need {len(self._needed_by)} sample(s), every "
            f"{frequency_to_string(self.frequency)}; check 'from.frequency' against what "
            "the source actually provides."
        )
