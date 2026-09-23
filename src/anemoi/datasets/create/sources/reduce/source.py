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
import logging
from typing import Any

from anemoi.transform import FieldList
from anemoi.utils.dates import frequency_to_string

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.sources import source_registry

from ..windowed.description import ReduceSchema
from ..windowed.description import check_window_inside_run
from ..windowed.description import validate_from
from ..windowed.description import window_samples
from ..windowed.description.instants import FromRun
from ..windowed.source import WindowSourceBase
from .plan import SamplingPlan
from .reducer import AverageReducer
from .reducer import MaximumReducer
from .reducer import MinimumReducer
from .reducer import SampleReducer
from .reducer import describe

LOG = logging.getLogger(__name__)


class ReduceSource(WindowSourceBase):
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
        Which metadata keys identify a variable; same meaning and defaults as in
        ``accumulate``.
    **kwargs : Any
        ``from:`` arrives here because ``from`` is a Python keyword.
    """

    schema = ReduceSchema

    #: The reduction this source performs.
    reducer_class: type[SampleReducer]

    def __init__(
        self,
        context: Any,
        source: Any,
        period: str | int | datetime.timedelta,
        group_by: dict | None = None,
        **kwargs: Any,
    ) -> None:
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
                f"{self.name}: 'from:' is required \u2014 state the cadence of the source data, "
                "e.g. 'from: {frequency: 6h}'"
            )

        # Validated through the same helper as the recipe schema, so recipe-time and
        # build-time validation cannot drift apart.  Set before super().__init__,
        # which needs the description to pick the MARS `type` default.
        self._from = validate_from(from_)

        super().__init__(context, source=source, period=period, group_by=group_by)

        # Raises when the window is not a whole number of samples.
        window_samples(datetime.datetime(2000, 1, 1), self.period, self.frequency)

    @property
    def frequency(self) -> datetime.timedelta:
        """The cadence of the source data (``from.frequency``)."""
        return self._from.frequency

    @property
    def is_run_anchored(self) -> bool:
        """Whether ``from:`` describes the run the trajectory layout imposes."""
        return isinstance(self._from, FromRun)

    def _mars_type_default(self) -> str:
        # A run-anchored description reads a forecast; a base-less one reads fields
        # indexed by validity time, i.e. an analysis.
        return "fc" if self.is_run_anchored else "an"

    def _hash_parts(self) -> tuple:
        return (str(self.frequency), self.is_run_anchored)

    def _plan(self) -> SamplingPlan:
        return SamplingPlan(
            period=self.period,
            frequency=self.frequency,
            reducer_class=self.reducer_class,
            run_anchored=self.is_run_anchored,
            name=self.name,
        )

    # ── dispatch branches ────────────────────────────────────────────

    def execute_valid_dates(self, dates: ValidDates) -> FieldList:
        """Reduce one window per requested validity date (gridded layout)."""
        if self.is_run_anchored:
            raise ValueError(
                f"{self.name}: 'from: {{base_dates: true, ...}}' inherits the run from the "
                "output layout, which only 'layout: trajectories' imposes. In any other "
                "layout describe base-less source data with 'from: {frequency: ...}'"
            )

        for d in dates:
            if not isinstance(d, datetime.datetime):
                raise TypeError(f"{self.name}: valid_date must be a datetime.datetime instance, got {type(d)}")

        return self._run([(d, None) for d in dates])

    def execute_forecast_dates(self, dates: ForecastDates) -> FieldList:
        """Reduce one window per ``(valid_time, basetime)`` row (trajectories layout).

        A base-less ``from:`` reads an analysis archive by validity time and the row's
        basetime only stamps the output; a run-anchored one reads lead times of the run
        the layout imposes.
        """
        targets = [(valid_time, basetime) for valid_time, basetime in dates.items]

        if self.is_run_anchored:
            # The window has to lie inside the run; a base-less source has no such
            # restriction (analyses exist before the basetime too).
            for valid_time, basetime in targets:
                check_window_inside_run(valid_time, basetime, self.period, self.name)

        return self._run(targets)

    def _run(self, targets: list[tuple]) -> FieldList:
        """Resolve the windows, fetch, reduce and check."""
        plan = self._plan()
        parts = plan.parts_for(targets)
        reducers, fields = self._reduce_fields(
            plan, self._create_source_object(), plan.argument(targets, parts), targets, parts
        )
        return self._finalise(reducers, fields, targets)

    def _finalise(self, reducers: dict, fields: list, targets: list[tuple]) -> FieldList:
        """Check that every window was complete and return the reduced fields."""
        if not reducers:
            raise ValueError(f"{self.name}: the source returned no usable field, cannot reduce anything")

        incomplete = {k: r for k, r in reducers.items() if not r.is_complete()}
        if incomplete:
            raise ValueError(
                f"{self.name}: {len(incomplete)} window(s) are missing source samples \u2014 a "
                "reduction over an incomplete window would silently bias the result and its "
                f"statistics:\n{describe(incomplete)}"
            )

        # A variable that is missing for a whole target creates no reducer at all, so
        # completeness alone would not catch it.
        keys = {key for *_, key in reducers}
        missing = [(t, key) for t in targets for key in sorted(keys) if (*t, key) not in reducers]
        if missing:
            detail = "\n".join(
                f"  {vdate}{f' (basetime {basetime})' if basetime is not None else ''}: {dict(key)}"
                for (vdate, basetime), key in missing[:20]
            )
            raise ValueError(
                f"{self.name}: no source data at all for {len(missing)} (date, variable) "
                f"combination(s):\n{detail}"
            )

        LOG.info("%s: created %d reduced fields over %s", self.name, len(fields), frequency_to_string(self.period))
        return self._as_fieldlist(fields)


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
