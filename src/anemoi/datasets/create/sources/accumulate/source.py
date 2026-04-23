# (C) Copyright 2025 Anemoi contributors.
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

import earthkit.data
from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.core.temporary import temp_file
from earthkit.data.readers.grib.output import new_grib_output

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.dispatch import for_forecast_dates
from anemoi.datasets.create.dispatch import for_valid_dates
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry

from .accumulator import Accumulator
from .accumulator import Logs
from .covering import ForecastCovering
from .covering import covering_factory
from .field_to_interval import FieldToInterval

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
            assert key in ignore, f"{key} absent in ignore list {ignore}, at least 'date', 'time', 'step' required"
        return group_by



@source_registry.register("accumulate")
class AccumulateSource(Source):

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
    ) -> None:
        super().__init__(context)

        if "accumulation_period" in source:
            raise ValueError("'accumulation_period' should be define outside source for accumulate action as 'period'")

        if availability is not None and covering is not None:
            raise ValueError(
                "Cannot specify both 'availability' (deprecated) and 'covering' " "in the same accumulate block."
            )
        if availability is not None:
            warnings.warn(
                "'availability:' is deprecated; use 'covering: { auto: <value> }' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            covering = {"auto": availability}

        self.source = source
        self.period = frequency_to_timedelta(period)
        self.covering = covering
        self.accumulation = accumulation
        self.patch = patch
        self.group_by = patch_groupby_keys(group_by)
        self._field_to_interval = FieldToInterval(patch)
        self._source_name = self._prepare_source()

    # ── shared helpers ───────────────────────────────────────────────

    def _prepare_source(self):
        """Validate source config and apply MARS defaults."""
        source = self.source
        assert isinstance(source, dict) and len(source) == 1, (
            f"Source must have exactly one key, got {list(source.keys())}"
        )
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
            json.dumps((str(self.period), self.source, *extra_hash_parts), sort_keys=True).encode()
        ).hexdigest()
        return self.context.create_source(self.source, "data_sources", h)

    def _extract_field_info(self, field):
        """Extract values, grouping key, time interval, and log string from a field."""
        values = field.values.copy()
        meta = field.metadata(namespace=self.group_by["namespace"])
        key = {k: v for k, v in meta.items() if k not in self.group_by["ignore"]}
        key = tuple(sorted(key.items()))
        log = " ".join(f"{k}={v}" for k, v in meta.items())
        field_interval = self._field_to_interval(field)
        return values, key, field_interval, log

    def _finalise(self, accumulators, output, tmp):
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

        output.close()
        ds = earthkit.data.from_source("file", tmp.path)
        ds._keep_file = tmp  # prevent deletion of temp file until ds is deleted

        LOG.debug(f"Created {len(ds)} accumulated fields:")
        for f in ds:
            LOG.debug("  %s", f)
        return ds

    def _accumulate_fields(self, source_object, intervals, targets, coverages):
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
        """
        # need a temporary file to store the accumulated fields for now, because earthkit-data
        # does not completely support in-memory fieldlists yet (metadata consistency is not fully ensured)
        tmp = temp_file()
        output = new_grib_output(tmp.path)

        accumulators = {}
        logs = Logs(
            accumulators=accumulators,
            source=self.source,
            source_object=source_object(self.context, intervals),
            field_to_interval=self._field_to_interval,
        )
        for field in source_object(self.context, intervals):
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
                        vdate, period=self.period, key=key,
                        coverage=coverages[target], basetime=basetime,
                    )

                acc = accumulators[accumulator_key]

                if acc.compute(values, field_interval):
                    # actual computation happened in this .compute() method
                    field_used = True
                    logs[-1][3].append(target)
                    logs[-1][4].append(acc.__repr__(verbose=True))

                    if acc.is_complete():
                        acc.write_to_output(output, template=field)

            if not field_used:
                logs.raise_error("Field not used for any accumulation", field=field, field_interval=field_interval)

        return accumulators, output, tmp

    # ── dispatch branches ────────────────────────────────────────────

    @for_valid_dates
    def execute(self, dates: ValidDates) -> Any:
        """Handle archive (validity-date) accumulations."""
        if self.covering is None:
            raise ValueError(
                "Argument 'covering' (or its deprecated alias 'availability') must be "
                "specified for accumulate source. See "
                "https://anemoi.readthedocs.io/projects/datasets/en/latest/building/sources/accumulate.html"
            )

        LOG.debug("💬 source for accumulations: %s", self.source)
        source_object = self._create_source_object()
        covering_obj = covering_factory(self.covering, self._source_name, self.source[self._source_name])

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

        accumulators, output, tmp = self._accumulate_fields(source_object, intervals, targets, coverages)

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

        return self._finalise(accumulators, output, tmp)

    @for_forecast_dates
    def execute(self, dates: ForecastDates) -> Any:
        """Handle forecast (trajectory) accumulations."""
        if self.accumulation is None:
            raise ValueError(
                "Argument 'accumulation' (one of 'from-zero', 'from-previous-step') "
                "is mandatory for accumulate sources used in trajectory recipes."
            )
        if self.covering is not None:
            LOG.debug("Trajectory branch: ignoring 'covering:' (basetime imposed by caller).")

        LOG.debug("💬 source for forecast accumulations: %s", self.source)
        source_object = self._create_source_object(self.accumulation)
        covering = ForecastCovering(period=self.period, accumulation=self.accumulation)

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

        accumulators, output, tmp = self._accumulate_fields(source_object, forecast_intervals, targets, coverages)

        return self._finalise(accumulators, output, tmp)
