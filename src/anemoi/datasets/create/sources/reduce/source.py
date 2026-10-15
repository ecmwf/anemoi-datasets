# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The windowed time-reduction sources: ``average``, ``minimum`` and ``maximum``.

They share :class:`ReduceSource`; each registered spelling only names the
reduction it performs.  There is deliberately no ``reduce:`` source and no
``operation:`` key in a recipe — the reduction is the verb, so it is the name
of the block.

The recipe keys mirror ``accumulate``: ``source:`` is where the data comes
from, ``period:`` is the window wanted, and ``from:`` is what the source data
is (see :mod:`.description`).

.. code:: yaml

   average:
     period: 24h
     from: {frequency: 6h}
     source: {mars: {class: ea, type: an, param: [2t], ...}}

``from:`` decides which of two loops runs. Instantaneous fields that *exist every*
``frequency`` are sampled inside the window, here. Fields that each *span* an interval
(``from: {accumulation: ...}`` or explicit step pairs) are reduced over the subwindows
covering the window, by the same loop as ``accumulate`` (:mod:`..windowed.ranges`).

Under a trajectory layout, ``from:`` also decides what the subsource is asked
for: a base-less ``{frequency: ...}`` is fetched by validity time and the row's
basetime only stamps the output, while ``{base_dates: true, frequency: ...}``
fetches lead times of the run the layout imposes.
"""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from typing import Any

from anemoi.transform import FieldList
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry

from ..windowed.check import Pieces
from ..windowed.check import check
from ..windowed.check import field_statistic
from ..windowed.common import check_complete
from ..windowed.common import create_source_object
from ..windowed.common import group_key
from ..windowed.common import prepare_source
from ..windowed.covering import ForecastCovering
from ..windowed.covering import covering_from_description
from ..windowed.description import FromInstants
from ..windowed.description import FromRun
from ..windowed.description import ReduceSchema
from ..windowed.description import check_window_inside_run
from ..windowed.description import takes_run_from_layout
from ..windowed.description import validate_from
from ..windowed.description import window_samples
from ..windowed.field_to_interval import FieldToInterval
from ..windowed.groupby import patch_groupby_keys
from ..windowed.operations import OPERATIONS
from ..windowed.ranges import argument
from ..windowed.ranges import cut
from ..windowed.ranges import reduce_ranges
from .reducer import SampleReducer

LOG = logging.getLogger(__name__)


def _valid_datetime(field) -> datetime.datetime:
    """The validity time of *field*.

    The earthkit time component knows it for every field shape; GRIB-backed
    fields that predate that component are read from ``validityDate`` /
    ``validityTime`` instead.
    """
    try:
        return field.time.valid_datetime()
    except AttributeError:
        date_str = str(field.metadata("validityDate")).zfill(8)
        time_str = str(field.metadata("validityTime")).zfill(4)
        return datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")


def _base_datetime(field) -> datetime.datetime:
    """The model-run base time of *field*.

    The mars ``date``/``time`` keys give the base time for ordinary forecasts,
    but for hindcast fields ``date`` is the reforecast reference date while the
    run starts at ``hdate``; the field's time component is right in both cases.
    """
    try:
        return field.time.base_datetime()
    except AttributeError:
        date_str = str(field.metadata("date")).zfill(8)
        time_str = str(field.metadata("time")).zfill(4)
        return datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M")


class ReduceSource(Source):
    """Reduce a window of source fields to one field per date.

    Not registered itself: the registered sources are :class:`AverageSource`,
    :class:`MinimumSource` and :class:`MaximumSource`, which differ only in
    :attr:`operation`.

    Parameters
    ----------
    context : Any
        The build context.
    source : dict
        The subsource, as a single-key dictionary (``{mars: {...}}``).
    period : str or int or datetime.timedelta
        The reduction window, e.g. ``24h``.
    over : str or int or datetime.timedelta, optional
        For interval-valued source data only: cut the window into ``over``-long
        subwindows, each rebuilt by a signed sum, so ``over: 1h`` under ``maximum:``
        is the largest hourly total of an accumulating archive.
    group_by : dict, optional
        Which metadata keys identify a variable; same meaning and defaults as
        in ``accumulate``.
    **kwargs : Any
        ``from:`` arrives here because ``from`` is a Python keyword.
    """

    schema = ReduceSchema

    #: The reduction this source performs: ``max``, ``min`` or ``mean``.
    operation: str

    #: The registered recipe spelling, for error messages.
    name: str

    def __init__(
        self,
        context: Any,
        source: Any,
        period: str | int | datetime.timedelta,
        over: str | int | datetime.timedelta | None = None,
        group_by: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(context)

        # `from` is a Python keyword, so it can only arrive through kwargs.
        # A raw recipe spells it `from:`; a recipe that has been through the
        # pydantic schema is dumped by field name and spells it `from_`.
        from_keys = [k for k in ("from", "from_") if k in kwargs]
        if len(from_keys) > 1:
            raise ValueError(f"{self.name}: specify 'from' once, not both 'from' and 'from_'")
        from_ = kwargs.pop(from_keys[0], None) if from_keys else None

        # Raw (non-pydantic-validated) configs may spell keys with hyphens.
        if "group-by" in kwargs:
            if group_by is not None:
                raise ValueError(f"{self.name}: cannot specify both 'group_by' and 'group-by'")
            group_by = kwargs.pop("group-by")
        if kwargs:
            raise TypeError(f"{self.name}: unknown argument(s) {sorted(kwargs)}")

        if from_ is None:
            raise ValueError(
                f"{self.name}: 'from:' is required — state the cadence of the source data, "
                "e.g. 'from: {frequency: 6h}'"
            )

        self._from = validate_from(from_)
        self.source = source
        self.period = frequency_to_timedelta(period)
        self.over = frequency_to_timedelta(over) if over is not None else None
        self.group_by = patch_groupby_keys(group_by, source_name=self.name)

        # Validated through the recipe schema, so recipe-time and build-time
        # validation cannot drift apart; it also owns every rule about `over:`.
        ReduceSchema.model_validate(
            {"period": self.period, "from": self._from, "source": self.source, "over": self.over}
        )

        # A run-anchored description reads a forecast; a base-less instantaneous one
        # reads fields indexed by validity time, i.e. an analysis; interval-valued
        # data is archived per step of a forecast.
        mars_type = "an" if self.is_instant_valued and not self.is_run_anchored else "fc"
        self._source_name = prepare_source(self.name, self.source, mars_type)

    @property
    def frequency(self) -> datetime.timedelta:
        """The cadence of the source data (``from.frequency``); instant-valued only."""
        return self._from.frequency

    @property
    def is_instant_valued(self) -> bool:
        """Whether ``from:`` describes fields that *exist every* frequency.

        The alternative is interval-valued source data -- fields that each *span* an
        interval -- reduced over the subwindows covering the window.
        """
        return isinstance(self._from, (FromInstants, FromRun))

    @property
    def is_run_anchored(self) -> bool:
        """Whether ``from:`` describes the run the trajectory layout imposes."""
        return takes_run_from_layout(self._from)

    # ── dispatch branches ────────────────────────────────────────────

    def execute_valid_dates(self, dates: ValidDates) -> FieldList:
        """Reduce one window per requested validity date (gridded layout)."""
        if self.is_run_anchored:
            raise ValueError(
                f"{self.name}: 'from: {{base_dates: true, ...}}' and 'base_dates: from-layout' "
                "take the run from the output layout, which only 'layout: trajectories' imposes. "
                "In any other layout describe the source data without it"
            )

        for d in dates:
            if not isinstance(d, datetime.datetime):
                raise TypeError(f"{self.name}: valid_date must be a datetime.datetime instance, got {type(d)}")

        targets = [(d, None) for d in dates]
        if not self.is_instant_valued:
            return self._reduce_ranges(targets)

        samples = self._window_samples(targets)
        wanted = sorted({sample for window in samples.values() for sample in window})
        return self._reduce_samples(ValidDates(wanted), targets, samples)

    def execute_forecast_dates(self, dates: ForecastDates) -> FieldList:
        """Reduce one window per ``(valid_time, basetime)`` row (trajectories layout).

        Two source shapes are served, and they differ only in what is asked of
        the subsource:

        - a **base-less** ``from: {frequency: ...}`` reads an analysis archive
          by validity time, and the row's basetime is used only to stamp the
          output;
        - a **run-anchored** ``from: {base_dates: true, ...}`` reads the
          run the layout imposes, so each sample is asked for as
          ``(sample_time, basetime)`` — one lead time of that run.

        Interval-valued source data is the same choice: ``base_dates:
        from-layout`` reads the run from the layout; any other description is
        covered by the same search as on the validity-date path, and the row's
        basetime only stamps the output.
        """
        targets = [(valid_time, basetime) for valid_time, basetime in dates.items]

        if self.is_run_anchored:
            # The window has to lie inside the run; a base-less source has no
            # such restriction (analyses exist before the basetime too).
            for valid_time, basetime in targets:
                check_window_inside_run(valid_time, basetime, self.period, self.name)

        if not self.is_instant_valued:
            return self._reduce_ranges(targets, from_layout=self.is_run_anchored)

        samples = self._window_samples(targets)

        if self.is_run_anchored:
            # Each sample belongs to the run of its own row, so the same
            # validity time reached from two runs stays two distinct requests.
            wanted = ForecastDates(
                sorted({(sample, basetime) for (_, basetime), window in samples.items() for sample in window})
            )
        else:
            wanted = ValidDates(sorted({sample for window in samples.values() for sample in window}))

        return self._reduce_samples(wanted, targets, samples)

    # ── interval-valued source data ──────────────────────────────────

    def _reduce_ranges(self, targets: list[tuple], from_layout: bool = False) -> FieldList:
        """Reduce the subwindows covering each window, with ``accumulate``'s loop.

        With ``from_layout`` the source data is the run of each row, so every window is
        covered from that run alone (no search) and the source is asked for
        ``ForecastIntervals``; otherwise the archive is searched.
        """
        if from_layout:
            covering = ForecastCovering(period=self.period, accumulation=self._from.accumulation)
        else:
            covering = covering_from_description(self._from, period=self.period, source_name=self.name)
        windows = {
            (vt, bt): cut(covering, vt - self.period, vt, basetime=bt if from_layout else None, over=self.over)
            for vt, bt in targets
        }

        key_parts = (self.name, str(self.period), self._from.model_dump_json(), str(self.over), self.source)
        source_object = create_source_object(self.context, self.source, key_parts)
        field_to_interval = FieldToInterval()

        reducers, fields = reduce_ranges(
            source_object(self.context, argument(targets, windows, self.period, forecast=from_layout)),
            targets=targets,
            windows=windows,
            period=self.period,
            operation=OPERATIONS[self.operation],
            group_by=self.group_by,
            field_to_interval=field_to_interval,
            source=self.source,
            stamp_basetime=True,
            # An `accumulation` scheme says the fields accumulate; explicit step pairs
            # (the way to describe a stored-extremum archive) say nothing.
            declared="accum" if getattr(self._from, "accumulation", None) is not None else None,
        )
        check_complete(self.name, self.period, reducers, fields, targets)
        return FieldList.from_fields(fields)

    # ── instantaneous source data ────────────────────────────────────

    def _window_samples(self, targets: list[tuple]) -> dict[tuple, list[datetime.datetime]]:
        """The sample times of every target's window, keyed by target.

        For a run-anchored description these are still *validity* times; the
        lead times they imply — ``step - period + k*frequency`` — are derived
        from them and the row's basetime, never declared.
        """
        return {target: window_samples(target[0], self.period, self.frequency) for target in targets}

    def _sample_id(self, valid_datetime: datetime.datetime, basetime: datetime.datetime | None) -> tuple:
        """The identity a sample is matched on.

        Base-less source data is identified by validity time alone.  Run-anchored
        source data is identified by ``(validity time, basetime)``: the same
        validity time reached from two different runs is two different fields
        and must never be reduced together.
        """
        return (valid_datetime, basetime if self.is_run_anchored else None)

    def _reduce_samples(self, argument: Any, targets: list[tuple], samples: dict) -> FieldList:
        """Fetch the samples and reduce each window down to one field.

        Parameters
        ----------
        argument : ValidDates or ForecastDates
            What to ask the subsource for.
        targets : list of tuple
            The ``(valid_date, basetime)`` rows to produce; *basetime* is
            ``None`` in the gridded layout.
        samples : dict
            Each target's window, as a list of sample validity times.

        Returns
        -------
        FieldList
            The reduced fields.
        """
        # The reverse index: which targets need a given sample.  Windows overlap
        # whenever `period` exceeds the output frequency, so one source field
        # commonly feeds several of them.
        needed_by: dict[tuple, list[tuple]] = defaultdict(list)
        for target in targets:
            basetime = target[1]
            for sample in samples[target]:
                needed_by[self._sample_id(sample, basetime)].append(target)

        LOG.debug(
            "%s: %d target(s) × %s / %s → %d source sample(s)",
            self.name,
            len(targets),
            frequency_to_string(self.period),
            frequency_to_string(self.frequency),
            len(needed_by),
        )

        key_parts = (self.name, str(self.period), str(self.frequency), self.is_run_anchored, self.source)
        source_object = create_source_object(self.context, self.source, key_parts)
        input_fields = source_object(self.context, argument)

        operation = OPERATIONS[self.operation]
        pieces = Pieces.of_samples(self.frequency, samples.values())
        checked: set = set()

        reducers: dict[tuple, SampleReducer] = {}
        fields = []

        for field in input_fields:
            valid_datetime = _valid_datetime(field)
            basetime = _base_datetime(field) if self.is_run_anchored else None
            key = group_key(field, self.group_by)
            values = field.values

            if key not in checked:
                check(operation.name, field_statistic(field), pieces)
                checked.add(key)

            used = False
            for target in needed_by.get(self._sample_id(valid_datetime, basetime), ()):
                reducer_key = (*target, key)
                if reducer_key not in reducers:
                    reducers[reducer_key] = SampleReducer(
                        target[0],
                        period=self.period,
                        key=key,
                        samples=samples[target],
                        operation=operation,
                        basetime=target[1],
                    )
                reducer = reducers[reducer_key]
                if reducer.compute(values, valid_datetime):
                    used = True
                    if reducer.is_complete():
                        fields.append(reducer.as_field(template=field))

            if not used:
                run = f" of the run based at {basetime}" if self.is_run_anchored else ""
                raise ValueError(
                    f"{self.name}: field {field} (valid {valid_datetime}{run}) is not part of "
                    f"any window. The windows need {len(needed_by)} sample(s), every "
                    f"{frequency_to_string(self.frequency)}; check 'from.frequency' against "
                    "what the source actually provides."
                )

        check_complete(self.name, self.period, reducers, fields, targets)
        return FieldList.from_fields(fields)


@source_registry.register("average")
class AverageSource(ReduceSource):
    """Time-average of source fields over ``period``."""

    name = "average"
    operation = "mean"


@source_registry.register("minimum")
class MinimumSource(ReduceSource):
    """Time-minimum of source fields over ``period``."""

    name = "minimum"
    operation = "min"


@source_registry.register("maximum")
class MaximumSource(ReduceSource):
    """Time-maximum of source fields over ``period``."""

    name = "maximum"
    operation = "max"
