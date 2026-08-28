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

Under a trajectory layout, ``from:`` also decides what the subsource is asked
for: a base-less ``{frequency: ...}`` is fetched by validity time and the row's
basetime only stamps the output, while ``{base_dates: true, frequency: ...}``
fetches lead times of the run the layout imposes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
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

from ..accumulate.source import patch_groupby_keys
from .description import FromRun
from .description import ReduceSchema
from .description import check_window_inside_run
from .description import validate_from
from .description import window_samples
from .reducer import AverageReducer
from .reducer import MaximumReducer
from .reducer import MinimumReducer
from .reducer import Reducer
from .reducer import describe

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
    """Reduce a window of instantaneous source fields to one field per date.

    Not registered itself: the registered sources are :class:`AverageSource`,
    :class:`MinimumSource` and :class:`MaximumSource`, which differ only in
    :attr:`reducer_class`.

    Parameters
    ----------
    context : Any
        The build context.
    source : dict
        The subsource, as a single-key dictionary (``{mars: {...}}``).
    period : str or int or datetime.timedelta
        The reduction window, e.g. ``24h``.
    group_by : dict, optional
        Which metadata keys identify a variable; same meaning and defaults as
        in ``accumulate``.
    **kwargs : Any
        ``from:`` arrives here because ``from`` is a Python keyword.
    """

    schema = ReduceSchema

    #: The reduction this source performs.
    reducer_class: type[Reducer]

    #: The registered recipe spelling, for error messages.
    name: str

    def __init__(
        self,
        context: Any,
        source: Any,
        period: str | int | datetime.timedelta,
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

        # Validated through the same helper as the recipe schema, so recipe-time
        # and build-time validation cannot drift apart.
        self._from = validate_from(from_)

        self.source = source
        self.period = frequency_to_timedelta(period)
        self.group_by = patch_groupby_keys(group_by, source_name=self.name)
        self._source_name = self._prepare_source()

        # Raises when the window is not a whole number of samples.
        window_samples(datetime.datetime(2000, 1, 1), self.period, self._from.frequency)

    @property
    def frequency(self) -> datetime.timedelta:
        """The cadence of the source data (``from.frequency``)."""
        return self._from.frequency

    @property
    def is_run_anchored(self) -> bool:
        """Whether ``from:`` describes the run the trajectory layout imposes."""
        return isinstance(self._from, FromRun)

    # ── shared helpers ───────────────────────────────────────────────

    def _prepare_source(self) -> str:
        """Validate the subsource config and apply MARS defaults."""
        source = self.source
        if not (isinstance(source, dict) and len(source) == 1):
            raise ValueError(f"{self.name}: 'source' must have exactly one key, got {sorted(source)}")
        source_name, source_config = next(iter(source.items()))
        if source_name == "mars":
            if "type" not in source_config:
                # A run-anchored description reads a forecast; a base-less one
                # reads fields indexed by validity time, i.e. an analysis.
                default = "fc" if self.is_run_anchored else "an"
                source_config["type"] = default
                LOG.warning(
                    f"{self.name}: assuming 'type: {default}' for the mars source as the recipe " "did not specify one"
                )
            if "levtype" not in source_config:
                source_config["levtype"] = "sfc"
                LOG.warning(
                    f"{self.name}: assuming 'levtype: sfc' for the mars source as the recipe did not specify one"
                )
        return source_name

    def _create_source_object(self):
        """Create a cached subsource object keyed by a content hash."""
        parts = (
            self.name,
            str(self.period),
            str(self.frequency),
            self.is_run_anchored,
            self.source,
        )
        h = hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
        return self.context.create_source(self.source, "data_sources", h)

    def _group_key(self, field) -> tuple:
        """The metadata identifying the variable a field belongs to."""
        meta = field.get(collections=f"metadata.{self.group_by['namespace']}")
        key = {k: v for k, v in meta.items() if k not in self.group_by["ignore"]}
        return tuple(sorted(key.items()))

    # ── dispatch branches ────────────────────────────────────────────

    def execute_valid_dates(self, dates: ValidDates) -> FieldList:
        """Reduce one window per requested validity date (gridded layout).

        Parameters
        ----------
        dates : ValidDates
            The output dates of the dataset being built.

        Returns
        -------
        FieldList
            One reduced field per date and per variable.
        """
        if self.is_run_anchored:
            raise ValueError(
                f"{self.name}: 'from: {{base_dates: true, ...}}' inherits the run from "
                "the output layout, which only 'layout: trajectories' imposes. In any other "
                "layout describe base-less source data with 'from: {frequency: ...}'"
            )

        for d in dates:
            if not isinstance(d, datetime.datetime):
                raise TypeError(f"{self.name}: valid_date must be a datetime.datetime instance, got {type(d)}")

        targets = [(d, None) for d in dates]
        samples = self._window_samples(targets)
        wanted = sorted({sample for window in samples.values() for sample in window})

        return self._reduce_fields(ValidDates(wanted), targets, samples)

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

        Parameters
        ----------
        dates : ForecastDates
            The ``(valid_time, basetime)`` rows of the trajectory being built.

        Returns
        -------
        FieldList
            One reduced field per row and per variable.
        """
        targets = [(valid_time, basetime) for valid_time, basetime in dates.items]

        if self.is_run_anchored:
            # The window has to lie inside the run; a base-less source has no
            # such restriction (analyses exist before the basetime too).
            for valid_time, basetime in targets:
                check_window_inside_run(valid_time, basetime, self.period, self.name)

        samples = self._window_samples(targets)

        if self.is_run_anchored:
            # Each sample belongs to the run of its own row, so the same
            # validity time reached from two runs stays two distinct requests.
            argument = ForecastDates(
                sorted({(sample, basetime) for (_, basetime), window in samples.items() for sample in window})
            )
        else:
            argument = ValidDates(sorted({sample for window in samples.values() for sample in window}))

        return self._reduce_fields(argument, targets, samples)

    # ── the shared reduction ─────────────────────────────────────────

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
        and must never be folded together.
        """
        return (valid_datetime, basetime if self.is_run_anchored else None)

    def _reduce_fields(self, argument: Any, targets: list[tuple], samples: dict) -> FieldList:
        """Fetch the samples and fold each window down to one field.

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

        source_object = self._create_source_object()
        input_fields = source_object(self.context, argument)

        reducers: dict[tuple, Reducer] = {}
        fields = []

        for field in input_fields:
            valid_datetime = _valid_datetime(field)
            basetime = _base_datetime(field) if self.is_run_anchored else None
            key = self._group_key(field)
            values = field.values

            used = False
            for target in needed_by.get(self._sample_id(valid_datetime, basetime), ()):
                reducer_key = (*target, key)
                if reducer_key not in reducers:
                    reducers[reducer_key] = self.reducer_class(
                        target[0],
                        period=self.period,
                        key=key,
                        samples=samples[target],
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

        return self._finalise(reducers, fields, targets)

    def _finalise(self, reducers: dict[tuple, Reducer], fields: list, targets: list[tuple]) -> FieldList:
        """Check that every window was complete and return the reduced fields."""
        if not reducers:
            raise ValueError(f"{self.name}: the source returned no usable field, cannot reduce anything")

        incomplete = {k: r for k, r in reducers.items() if not r.is_complete()}
        if incomplete:
            raise ValueError(
                f"{self.name}: {len(incomplete)} window(s) are missing source samples — a "
                "reduction over an incomplete window would silently bias the result and its "
                f"statistics:\n{describe(incomplete)}"
            )

        # A variable that is missing for a whole target creates no reducer at
        # all, so completeness alone would not catch it.
        keys = {key for *_, key in reducers}
        missing = [(t, key) for t in targets for key in sorted(keys) if (*t, key) not in reducers]
        if missing:
            detail = "\n".join(
                f"  {vdate}{f' (basetime {basetime})' if basetime is not None else ''}: {dict(key)}"
                for (vdate, basetime), key in missing[:20]
            )
            raise ValueError(
                f"{self.name}: no source data at all for {len(missing)} (date, variable) " f"combination(s):\n{detail}"
            )

        LOG.info("%s: created %d reduced fields over %s", self.name, len(fields), frequency_to_string(self.period))
        return FieldList.from_fields(fields)


@source_registry.register("average")
class AverageSource(ReduceSource):
    """Time-average of instantaneous source fields over ``period``."""

    name = "average"
    reducer_class = AverageReducer


@source_registry.register("minimum")
class MinimumSource(ReduceSource):
    """Time-minimum of instantaneous source fields over ``period``."""

    name = "minimum"
    reducer_class = MinimumReducer


@source_registry.register("maximum")
class MaximumSource(ReduceSource):
    """Time-maximum of instantaneous source fields over ``period``."""

    name = "maximum"
    reducer_class = MaximumReducer
