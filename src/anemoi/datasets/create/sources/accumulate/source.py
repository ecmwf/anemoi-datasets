# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import json
import logging
from typing import Any

from anemoi.transform import FieldList

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.sources import source_registry

from ..windowed.clip import apply_clip
from ..windowed.clip import normalise_clip
from ..windowed.covering import AutoCovering
from ..windowed.covering import ForecastCovering
from ..windowed.covering import ValidTimeCovering
from ..windowed.covering import covering_factory
from ..windowed.covering import covering_from_description
from ..windowed.description import AccumulateSchema
from ..windowed.description import FromBare
from ..windowed.description import FromLookupTable
from ..windowed.description import FromTrajectories
from ..windowed.description import TrajectoryIntervalGenerator
from ..windowed.description import check_valid_time_source
from ..windowed.description import infer_from_trajectories
from ..windowed.description import normalise_from
from ..windowed.field_to_interval import FieldToInterval
from ..windowed.interval_generators import LookupTableIntervalGenerator
from ..windowed.source import WindowSourceBase
from .plan import IntervalPlan

LOG = logging.getLogger(__name__)

# TODO:
# for od-oper: need to do this adjustment, should be in mars source itself?
# Modifies the request stream based on the time (so, not here).
# if request["time"] in (6, 18, 600, 1800):
#    request["stream"] = "scda"
# else:
#    request["stream"] = "oper"


@source_registry.register("accumulate")
class AccumulateSource(WindowSourceBase):

    name = "accumulate"
    schema = AccumulateSchema

    def __init__(
        self,
        context: Any,
        source: Any,
        period: str | int | datetime.timedelta,
        availability=None,
        covering=None,
        accumulation: str | None = None,
        patch: Any = None,
        group_by: dict | None = None,
        clip: Any = None,
        **kwargs: Any,
    ) -> None:
        # `from` is a Python keyword, so it can only arrive through kwargs.
        # A raw recipe spells it `from:`; a recipe that has been through the
        # pydantic schema is dumped by field name and spells it `from_`.
        from_keys = [k for k in ("from", "from_") if k in kwargs]
        if len(from_keys) > 1:
            raise ValueError("accumulate: specify 'from' once, not both 'from' and 'from_'")
        from_ = kwargs.pop(from_keys[0], None) if from_keys else None

        # Raw (non-pydantic-validated) configs may spell keys with hyphens;
        # accept both spellings.
        def _pop_hyphenated(name: str, value: Any) -> Any:
            alias = name.replace("_", "-")
            if alias in kwargs:
                if value is not None:
                    raise ValueError(f"accumulate: cannot specify both '{name}' and '{alias}'")
                value = kwargs.pop(alias)
            return value

        group_by = _pop_hyphenated("group_by", group_by)
        if "over" in kwargs:
            raise ValueError(
                "accumulate: 'over:' does not apply to a sum. It states how long each "
                "subwindow is, which matters only when the reduction differs from what a "
                "subwindow carries -- and a sum of subwindows is the same whatever length "
                "they are. Use it on 'maximum:'/'minimum:'/'average:'"
            )
        if kwargs:
            raise TypeError(f"accumulate: unknown argument(s) {sorted(kwargs)}")

        if "accumulation_period" in source:
            raise ValueError("'accumulation_period' should be define outside source for accumulate action as 'period'")

        # ── fold every spelling into `from:` (shared with the schema) ────
        # warn=False: deprecation warnings are a recipe-validation concern;
        # by the time the source is built the schema has already warned once.
        self._from, self.covering = normalise_from(
            from_=from_,
            accumulation=accumulation,
            covering=covering,
            availability=availability,
            warn=False,
        )

        self.patch = patch
        self.clip = normalise_clip(clip)
        self._field_to_interval = FieldToInterval(patch)

        super().__init__(context, source=source, period=period, group_by=group_by)

    def _finalise(self, reducers, fields):
        """Clean empty reducers, validate completeness, and return the dataset."""
        # some reducers may be empty, remove them
        # this can happen when the source provides fields that not exactly the one requested (scda/oper)
        empty = [k for k, reducer in reducers.items() if reducer.values is None]
        for k in empty:
            LOG.warning(f"Removing empty accumulator for key {k}")
            del reducers[k]

        for reducer in reducers.values():
            if not reducer.is_complete():
                raise ValueError(f"Reducer not complete: {reducer.__repr__(verbose=True)}")

        LOG.info(f"Created {len(reducers)} accumulated fields")

        if not reducers:
            raise ValueError("No reducers were created, cannot produce accumulated datasource")

        ds = apply_clip(FieldList.from_fields(fields), self.clip)

        LOG.debug(f"Created {len(ds)} accumulated fields:")
        for f in ds:
            LOG.debug("  %s", f)
        return ds

    # ── dispatch branches ────────────────────────────────────────────

    def _resolved_from(self):
        """The ``from:`` description, recognising it from the source when omitted.

        An omitted ``from:`` (``self._from is None``) with no legacy covering means
        "recognise the source data from the source" — resolved here against the
        (well-known MARS) source. When a legacy ``covering:`` is present ``from:``
        is ``None`` too, but the covering owns the description, so it is returned
        unchanged for the legacy branch.
        """
        if self._from is None and self.covering is None:
            description = infer_from_trajectories(self._source_name, self.source[self._source_name])
            LOG.info("from: (omitted) recognised as: %s", description.model_dump(mode="json"))
            return description
        return self._from

    def _searched_covering(self):
        """Build the Covering for the validity-date path from the description."""
        description = self._resolved_from()
        if description is not None:
            return covering_from_description(description, period=self.period)

        # Deprecated 'covering:'/'availability:' -- the legacy machinery.
        return covering_factory(self.covering, self._source_name, self.source[self._source_name])

    def _description_hash_part(self) -> str:
        """A stable string identifying the source-data description, for the source cache key."""
        if self._from is None:
            if self.covering is not None:
                return f"covering:{json.dumps(self.covering, sort_keys=True, default=str)}"
            # Omitted `from:` — recognised from the source at build time.
            return "from:recognise-from-source"
        return f"from:{self._from.model_dump_json()}"

    def execute_valid_dates(self, dates: ValidDates) -> Any:
        """Handle validity-date accumulations.

        An omitted ``from:`` with no legacy covering is not an error — it means
        the source data is recognised from the source (see :meth:`_resolved_from`);
        recognition of a non-well-known source fails loudly at that point.
        """
        LOG.debug("💬 source for accumulations: %s", self.source)
        for d in dates:
            if not isinstance(d, datetime.datetime):
                raise TypeError("valid_date must be a datetime.datetime instance")

        plan = IntervalPlan(
            period=self.period,
            covering=self._searched_covering(),
            source=self.source,
            field_to_interval=self._field_to_interval,
        )
        targets = [(d, None) for d in dates]
        reducers, fields = self._run(plan, targets, self._description_hash_part())

        # Final checks
        for date in dates:
            count = sum(1 for k in reducers.keys() if k[0] == date)
            LOG.debug(f"Date {date} has {count} reducers")
            if count != len(reducers) // len(dates):
                LOG.error(f"All requested dates: {dates}")
                LOG.error(f"Date {date} has {count} reducers, expected {len(reducers) // len(dates)}")
                for k in reducers.keys():
                    if k[0] == date:
                        LOG.error(f"  Reducer for key {k}")
                raise ValueError(f"Date {date} has {count} reducers, expected {len(reducers) // len(dates)}")

        return self._finalise(reducers, fields)

    def execute_forecast_dates(self, dates: ForecastDates) -> Any:
        """Handle forecast (trajectory) accumulations.

        ``from:`` describes the subsource; the trajectory *output* is decided
        by the layout — the two are orthogonal, so the subsource is resolved
        exactly as in the validity-date path and only the output stamping
        differs.  There are two families:

        - ``from-layout`` :class:`FromTrajectories` — the subsource *is* the
          run the layout imposes, so the covering is the basetime-anchored
          :class:`ForecastCovering` (no search over the archive).
        - every other subsource (a bare, base-less valid-time source; an
          explicit-grid or recognised (omitted ``from:``) trajectory archive;
          a ``lookup-table``) —
          reconstructed by the same base-less covering *search* as
          :meth:`execute_valid_dates`; only the result is stamped as a forecast
          field at ``(basetime, step)``.
        """
        LOG.debug("💬 source for forecast accumulations: %s", self.source)
        description = self._resolved_from()

        if isinstance(description, FromTrajectories) and description.is_layout_grid:
            return self._execute_forecast_from_layout(dates, description)

        return self._execute_forecast_reconstructed(dates)

    def _execute_forecast_from_layout(self, dates: ForecastDates, description: FromTrajectories) -> Any:
        """Forecast accumulations for a ``from-layout`` subsource (the layout's own run).

        The layout imposes the basetime per row, so the covering is the trivial
        signed decomposition of :class:`ForecastCovering` — no search over the
        source data.
        """
        items = list(dates.items)
        plan = IntervalPlan(
            period=self.period,
            covering=ForecastCovering(period=self.period, accumulation=description.accumulation),
            source=self.source,
            field_to_interval=self._field_to_interval,
            basetime=True,
            forecast_items=items,
        )
        reducers, fields = self._run(plan, list(items), description.accumulation)
        return self._finalise(reducers, fields)

    def _execute_forecast_reconstructed(self, dates: ForecastDates) -> Any:
        """Forecast accumulations reconstructed from a searched subsource covering.

        Shares the covering *search* of :meth:`execute_valid_dates` (via
        :meth:`_searched_covering`): each output window ``[vt − period, vt]`` is
        covered from the subsource independently of the output basetime.  The
        covering intervals carry the subsource's own base (``None`` for a
        base-less valid-time source, the archive run for a trajectory archive),
        so the inner source fetches them unchanged; the accumulated result is
        stamped as a forecast field at the layout's ``(basetime, step)`` because
        the accumulator is given that basetime.
        """
        plan = IntervalPlan(
            period=self.period,
            covering=self._searched_covering(),
            source=self.source,
            field_to_interval=self._field_to_interval,
            basetime=True,
        )
        targets = list(dates.items)
        reducers, fields = self._run(plan, targets, self._description_hash_part())
        return self._finalise(reducers, fields)

    def _run(self, plan: IntervalPlan, targets: list, *hash_parts: Any) -> tuple:
        """Resolve the coverings, fetch and reduce."""
        parts = plan.parts_for(targets)
        source_object = self._create_source_object(*hash_parts)
        return self._reduce_fields(plan, source_object, plan.argument(targets, parts), targets, parts)
