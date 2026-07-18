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
import warnings
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
from .covering import covering_factory
from .description import ACCUMULATED_VALUES
from .description import DESCRIPTION_KEYS
from .description import MIGRATE_HINT
from .description import AccumulateSchema
from .description import FromTrajectories
from .description import TrajectoryIntervalGenerator
from .description import infer_from_trajectories
from .field_to_interval import FieldToInterval
from .interval_generators import CycleIntervalProvider
from .interval_generators import increments_generator

LOG = logging.getLogger(__name__)

# TODO:
# for od-oper: need to do this adjustment, should be in mars source itself?
# Modifies the request stream based on the time (so, not here).
# if request["time"] in (6, 18, 600, 1800):
#    request["stream"] = "scda"
# else:
#    request["stream"] = "oper"


def patch_groupby_keys(group_by: dict | None = None):
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
                    f"accumulate group_by: '{key}' absent in ignore list {ignore}; "
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
        accumulated: str | None = None,
        from_trajectories: Any = None,
        from_increments: str | int | datetime.timedelta | None = None,
        from_lookup_table: dict | None = None,
        patch: Any = None,
        group_by: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(context)

        # Raw (non-pydantic-validated) configs spell the description keys with
        # hyphens; accept both spellings.
        def _pop_hyphenated(name: str, value: Any) -> Any:
            alias = name.replace("_", "-")
            if alias in kwargs:
                if value is not None:
                    raise ValueError(f"accumulate: cannot specify both '{name}' and '{alias}'")
                value = kwargs.pop(alias)
            return value

        from_trajectories = _pop_hyphenated("from_trajectories", from_trajectories)
        from_increments = _pop_hyphenated("from_increments", from_increments)
        from_lookup_table = _pop_hyphenated("from_lookup_table", from_lookup_table)
        group_by = _pop_hyphenated("group_by", group_by)
        if kwargs:
            raise TypeError(f"accumulate: unknown argument(s) {sorted(kwargs)}")

        if "accumulation_period" in source:
            raise ValueError("'accumulation_period' should be define outside source for accumulate action as 'period'")

        # ── deprecated spellings (kept for one release) ──────────────────
        if accumulation is not None:
            if accumulated is not None:
                raise ValueError("Cannot specify both 'accumulated' and its deprecated alias 'accumulation'.")
            warnings.warn(
                f"'accumulation:' is deprecated; use 'accumulated:' instead ({MIGRATE_HINT}).",
                DeprecationWarning,
                stacklevel=2,
            )
            accumulated = accumulation

        if availability is not None and covering is not None:
            raise ValueError(
                "Cannot specify both 'availability' (deprecated) and 'covering' " "in the same accumulate block."
            )
        if availability is not None:
            covering = {"auto": availability}
        if covering is not None:
            warnings.warn(
                "'covering:'/'availability:' are deprecated; describe the archive with "
                "'from-trajectories:', 'from-increments:' or 'from-lookup-table:' instead "
                f"({MIGRATE_HINT}).",
                DeprecationWarning,
                stacklevel=2,
            )

        # ── exactly one archive description ──────────────────────────────
        descriptions = {
            "from-trajectories": from_trajectories,
            "from-increments": from_increments,
            "from-lookup-table": from_lookup_table,
            "covering": covering,
        }
        given = [k for k, v in descriptions.items() if v is not None]
        if len(given) > 1:
            raise ValueError(f"accumulate: only one archive description is allowed, got {given}")
        self._description_key = given[0] if given else None

        if accumulated is not None and self._description_key is not None:
            raise ValueError(
                f"accumulate: '{self._description_key}' and block-level 'accumulated:' are "
                "mutually exclusive — in archive recipes 'accumulated:' belongs inside "
                "'from-trajectories:'; bare 'accumulated:' is for trajectory-layout recipes"
            )

        if from_trajectories is not None and from_trajectories != "auto":
            if not isinstance(from_trajectories, FromTrajectories):
                from_trajectories = FromTrajectories.model_validate(from_trajectories)
        if from_increments is not None:
            from_increments = frequency_to_timedelta(from_increments)

        self.source = source
        self.period = frequency_to_timedelta(period)
        self.covering = covering
        self.accumulated = accumulated
        self.from_trajectories = from_trajectories
        self.from_increments = from_increments
        self.from_lookup_table = from_lookup_table
        self.patch = patch
        self.group_by = patch_groupby_keys(group_by)
        self._field_to_interval = FieldToInterval(patch)
        self._source_name = self._prepare_source()

        if from_increments is not None:
            if self.period % from_increments != datetime.timedelta(0):
                raise ValueError(
                    f"accumulate: 'from-increments' ({from_increments}) must divide 'period' ({self.period})"
                )
            if self._source_name in ("mars", "fdb"):
                raise ValueError(
                    f"accumulate: 'from-increments' describes a base-less, valid-time-indexed "
                    f"archive, but '{self._source_name}' fields are anchored to a model run — "
                    "use 'from-trajectories:' instead"
                )

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

            if field_interval.end <= field_interval.start:
                logs.raise_error("Invalid field interval with end <= start", field=field, field_interval=field_interval)

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

    def _archive_covering(self):
        """Build the Covering for the archive (validity-date) path from the description."""
        if self._description_key == "from-trajectories":
            description = self.from_trajectories
            if description == "auto":
                description = infer_from_trajectories(self._source_name, self.source[self._source_name])
                LOG.info("from-trajectories: auto resolved to: %s", description.model_dump(mode="json"))
            return AutoCovering(TrajectoryIntervalGenerator(description))

        if self._description_key == "from-increments":
            return AutoCovering(increments_generator(self.from_increments))

        if self._description_key == "from-lookup-table":
            return AutoCovering(CycleIntervalProvider(**self.from_lookup_table))

        # Deprecated 'covering:'/'availability:' — the legacy machinery.
        return covering_factory(self.covering, self._source_name, self.source[self._source_name])

    def _description_hash_part(self) -> str:
        """A stable string identifying the archive description, for the source cache key."""
        description = {
            "from-trajectories": self.from_trajectories,
            "from-increments": self.from_increments,
            "from-lookup-table": self.from_lookup_table,
            "covering": self.covering,
        }.get(self._description_key)
        if isinstance(description, FromTrajectories):
            return f"{self._description_key}:{description.model_dump_json()}"
        return f"{self._description_key}:{json.dumps(description, sort_keys=True, default=str)}"

    def execute_valid_dates(self, dates: ValidDates) -> Any:
        """Handle archive (validity-date) accumulations."""
        if self._description_key is None:
            raise ValueError(
                "accumulate: describe the archive with exactly one of "
                f"{', '.join(repr(k) for k in DESCRIPTION_KEYS)}. See "
                "https://anemoi.readthedocs.io/projects/datasets/en/latest/building/sources/accumulate.html"
            )

        LOG.debug("💬 source for accumulations: %s", self.source)
        source_object = self._create_source_object(self._description_hash_part())
        covering_obj = self._archive_covering()

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
        """Handle forecast (trajectory) accumulations."""
        if self.accumulated is None:
            raise ValueError(
                f"Argument 'accumulated' (one of {ACCUMULATED_VALUES}) "
                "is mandatory for accumulate sources used in trajectory recipes."
            )
        if self._description_key is not None:
            raise ValueError(
                f"accumulate: '{self._description_key}' is not allowed in a trajectory-layout "
                "recipe — the layout imposes the basetime; remove the archive description "
                "and declare 'accumulated:' on the accumulate block instead."
            )

        LOG.debug("💬 source for forecast accumulations: %s", self.source)
        source_object = self._create_source_object(self.accumulated)
        covering = ForecastCovering(period=self.period, accumulation=self.accumulated)

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
