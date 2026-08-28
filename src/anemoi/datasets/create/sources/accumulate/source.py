# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import hashlib
import json
import logging
from typing import Any

from anemoi.transform import FieldList
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry

from .accumulator import Accumulator
from .accumulator import Logs
from .covering import AutoCovering
from .covering import ForecastCovering
from .covering import ValidTimeCovering
from .covering import covering_factory
from .description import AccumulateSchema
from .description import FromBare
from .description import FromLookupTable
from .description import FromTrajectories
from .description import TrajectoryIntervalGenerator
from .description import check_valid_time_source
from .description import infer_from_trajectories
from .description import normalise_from
from .field_to_interval import FieldToInterval
from .interval_generators import LookupTableIntervalGenerator

LOG = logging.getLogger(__name__)

# TODO:
# for od-oper: need to do this adjustment, should be in mars source itself?
# Modifies the request stream based on the time (so, not here).
# if request["time"] in (6, 18, 600, 1800):
#    request["stream"] = "scda"
# else:
#    request["stream"] = "oper"


def patch_groupby_keys(group_by: dict | None = None, *, source_name: str = "accumulate"):
    """Validate a recipe ``group_by:`` block, filling in the default.

    Shared with the time-reduction sources (``average``/``minimum``/``maximum``),
    which use the same key with the same meaning; *source_name* only names the
    caller in the error messages.
    """
    if group_by is None:
        return {"namespace": "mars", "ignore": ["date", "time", "step"]}
    else:
        namespace = group_by.get("namespace", None)
        if namespace is None:
            raise ValueError("No namespace in group_by (set namespace: mars for default)")
        if namespace != "mars":
            raise ValueError(f"Namespace {namespace} not supported, use 'mars'")
        ignore = group_by.get("ignore", [])
        for key in ["date", "time", "step"]:
            if key not in ignore:
                raise ValueError(
                    f"{source_name} group_by: '{key}' absent in ignore list {ignore}; "
                    "at least 'date', 'time', 'step' are required"
                )
        return group_by


@source_registry.register("accumulate")
class AccumulateSource(Source):

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
        **kwargs: Any,
    ) -> None:
        super().__init__(context)

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

        self.source = source
        self.period = frequency_to_timedelta(period)
        self.patch = patch
        self.group_by = patch_groupby_keys(group_by)
        self._field_to_interval = FieldToInterval(patch)
        self._source_name = self._prepare_source()

    # ── shared helpers ───────────────────────────────────────────────

    def _prepare_source(self):
        """Validate source config and apply MARS defaults."""
        source = self.source
        assert (
            isinstance(source, dict) and len(source) == 1
        ), f"Source must have exactly one key, got {list(source.keys())}"
        source_name, source_config = next(iter(source.items()))
        if source_name == "mars":
            if "type" not in source_config:
                source_config["type"] = "fc"
                LOG.warning("Assuming 'type: fc' for mars source as it was not specified in the recipe")
            if "levtype" not in source_config:
                source_config["levtype"] = "sfc"
                LOG.warning("Assuming 'levtype: sfc' for mars source as it was not specified in the recipe")
        return source_name

    def _create_source_object(self, *extra_hash_parts):
        """Create a cached source object keyed by content hash."""
        h = hashlib.md5(
            json.dumps((str(self.period), self.source, *extra_hash_parts), sort_keys=True, default=str).encode()
        ).hexdigest()
        return self.context.create_source(self.source, "data_sources", h)

    def _extract_field_info(self, field):
        """Extract values, grouping key, time interval, and log string from a field."""
        values = field.values.copy()
        meta = field.get(collections=f"metadata.{self.group_by['namespace']}")
        key = {k: v for k, v in meta.items() if k not in self.group_by["ignore"]}
        key = tuple(sorted(key.items()))
        log = " ".join(f"{k}={v}" for k, v in meta.items())
        field_interval = self._field_to_interval(field)
        return values, key, field_interval, log

    def _finalise(self, accumulators, fields):
        """Clean empty accumulators, validate completeness, and return the dataset."""
        # some accumulators may be empty, remove them
        # this can happen when the source provides fields that not exactly the one requested (scda/oper)
        empty = [k for k, acc in accumulators.items() if acc.values is None]
        for k in empty:
            LOG.warning(f"Removing empty accumulator for key {k}")
            del accumulators[k]

        for acc in accumulators.values():
            if not acc.is_complete():
                raise ValueError(f"Accumulator not complete: {acc.__repr__(verbose=True)}")

        LOG.info(f"Created {len(accumulators)} accumulated fields")

        if not accumulators:
            raise ValueError("No accumulators were created, cannot produce accumulated datasource")

        ds = FieldList.from_fields(fields)

        LOG.debug(f"Created {len(ds)} accumulated fields:")
        for f in ds:
            LOG.debug("  %s", f)
        return ds

    def _accumulate_fields(self, source_object, intervals, targets, coverages) -> tuple:
        """Process fields from source and fill accumulators.

        Parameters
        ----------
        source_object
            Source factory callable (called as ``source_object(context, intervals)``).
        intervals
            ``Intervals`` or ``ForecastIntervals`` to pass to *source_object*.
        targets
            List of ``(vdate, basetime)`` tuples.  For the valid-date path
            *basetime* is ``None``.
        coverages
            Dict mapping each target tuple to its list of covering intervals.

        Returns
        -------
        tuple
            ``(accumulators, fields)``.
        """
        fields = []
        accumulators = {}
        # Execute the inner source once; the same FieldList feeds the main
        # loop and, on failure, the diagnostic dump in Logs.
        input_fields = source_object(self.context, intervals)
        logs = Logs(
            accumulators=accumulators,
            source=self.source,
            source_object=input_fields,
            field_to_interval=self._field_to_interval,
        )
        for field in input_fields:
            # for each field provided by the catalogue, find which accumulators need it and perform accumulation
            values, key, field_interval, log = self._extract_field_info(field)
            logs.append([str(field), log, field_interval, [], []])

            field_used = False
            for target in targets:
                # The target defines the accumulation we want to produce,
                # A target is a tuple:
                #    - (validity_date, None) for valid-date accumulations
                #    - (validity_date, basetime) for forecast accumulations (trajectories)
                # The covering intervals coverage[target] defines which intervals are needed.
                vdate, basetime = target
                accumulator_key = (*target, key)
                if accumulator_key not in accumulators:
                    accumulators[accumulator_key] = Accumulator(
                        vdate,
                        period=self.period,
                        key=key,
                        coverage=coverages[target],
                        basetime=basetime,
                    )

                acc = accumulators[accumulator_key]

                if acc.compute(values, field_interval):
                    # actual computation happened in this .compute() method
                    field_used = True
                    logs[-1][3].append(target)
                    logs[-1][4].append(acc.__repr__(verbose=True))

                    if acc.is_complete():
                        fields.append(acc.as_field(template=field))

            if not field_used:
                logs.raise_error("Field not used for any accumulation", field=field, field_interval=field_interval)

        return accumulators, fields

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

        if isinstance(description, FromTrajectories):
            return AutoCovering(TrajectoryIntervalGenerator(description))

        if isinstance(description, FromBare):
            # A bare `from:` is base-less, validity-time-indexed source data;
            # `accumulation` is a duration. The window is tiled directly (no
            # search, no midnight alignment), so the length need not divide 24h.
            check_valid_time_source(description, period=self.period)
            return ValidTimeCovering(description.duration)

        if isinstance(description, FromLookupTable):
            return AutoCovering(LookupTableIntervalGenerator(**description.entries()))

        # Deprecated 'covering:'/'availability:' — the legacy machinery.
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
        source_object = self._create_source_object(self._description_hash_part())
        covering_obj = self._searched_covering()

        # generate the interval coverage for every date
        coverages = {}
        for d in dates:
            if not isinstance(d, datetime.datetime):
                raise TypeError("valid_date must be a datetime.datetime instance")
            coverages[(d, None)] = covering_obj.cover(d - self.period, d)
            LOG.debug(f"  Found covering intervals: for {d - self.period} to {d}:")
            for c in coverages[(d, None)]:
                LOG.debug(f"    {c}")

        intervals = Intervals(dates, [i for d in dates for i in coverages[(d, None)]])
        targets = [(d, None) for d in dates]

        accumulators, fields = self._accumulate_fields(source_object, intervals, targets, coverages)

        # Final checks
        for date in dates:
            count = sum(1 for k in accumulators.keys() if k[0] == date)
            LOG.debug(f"Date {date} has {count} accumulators")
            if count != len(accumulators) // len(dates):
                LOG.error(f"All requested dates: {dates}")
                LOG.error(f"Date {date} has {count} accumulators, expected {len(accumulators) // len(dates)}")
                for k in accumulators.keys():
                    if k[0] == date:
                        LOG.error(f"  Accumulator for key {k}")
                raise ValueError(f"Date {date} has {count} accumulators, expected {len(accumulators) // len(dates)}")

        return self._finalise(accumulators, fields)

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
        source_object = self._create_source_object(description.accumulation)
        covering = ForecastCovering(period=self.period, accumulation=description.accumulation)

        coverages: dict = {}
        for vt, bt in dates.items:
            coverages[(vt, bt)] = covering.cover(vt - self.period, vt, basetime=bt)
            LOG.debug("  Forecast covering for (vt=%s, bt=%s):", vt, bt)
            for c in coverages[(vt, bt)]:
                LOG.debug("    %s", c)

        forecast_intervals = ForecastIntervals(
            items=[(vt, bt, self.period) for vt, bt in dates.items],
            intervals=[i for vt, bt in dates.items for i in coverages[(vt, bt)]],
        )
        targets = [(vt, bt) for vt, bt in dates.items]

        accumulators, fields = self._accumulate_fields(source_object, forecast_intervals, targets, coverages)

        return self._finalise(accumulators, fields)

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
        source_object = self._create_source_object(self._description_hash_part())
        covering_obj = self._searched_covering()

        items = list(dates.items)
        coverages: dict = {}
        for vt, bt in items:
            coverages[(vt, bt)] = covering_obj.cover(vt - self.period, vt)
            LOG.debug("  Reconstructed covering for (vt=%s, bt=%s):", vt, bt)
            for c in coverages[(vt, bt)]:
                LOG.debug("    %s", c)

        # Overlapping trajectory rows can request the same subsource window more
        # than once; fetch each interval once (matching remaps it to every row).
        seen: dict = {}
        for vt, bt in items:
            for i in coverages[(vt, bt)]:
                seen.setdefault(i, None)
        intervals = Intervals(dates=sorted({vt for vt, _ in items}), intervals=list(seen))
        targets = [(vt, bt) for vt, bt in items]

        accumulators, fields = self._accumulate_fields(source_object, intervals, targets, coverages)

        return self._finalise(accumulators, fields)
