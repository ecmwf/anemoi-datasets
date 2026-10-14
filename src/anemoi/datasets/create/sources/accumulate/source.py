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
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry

from ..windowed.clip import apply_clip
from ..windowed.clip import normalise_clip
from ..windowed.common import check_complete
from ..windowed.common import create_source_object
from ..windowed.common import prepare_source
from ..windowed.covering import ForecastCovering
from ..windowed.covering import covering_factory
from ..windowed.covering import covering_from_description
from ..windowed.description import AccumulateSchema
from ..windowed.description import FromTrajectories
from ..windowed.description import infer_from_trajectories
from ..windowed.description import normalise_from
from ..windowed.field_to_interval import FieldToInterval
from ..windowed.groupby import patch_groupby_keys
from ..windowed.operations import OPERATIONS
from ..windowed.ranges import argument
from ..windowed.ranges import cut
from ..windowed.ranges import reduce_ranges

LOG = logging.getLogger(__name__)

# TODO:
# for od-oper: need to do this adjustment, should be in mars source itself?
# Modifies the request stream based on the time (so, not here).
# if request["time"] in (6, 18, 600, 1800):
#    request["stream"] = "scda"
# else:
#    request["stream"] = "oper"


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
        clip: Any = None,
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
        self.clip = normalise_clip(clip)
        self._field_to_interval = FieldToInterval(patch)
        self._source_name = self._prepare_source()

    def _prepare_source(self) -> str:
        return prepare_source("accumulate", self.source, mars_type="fc")

    # ── the reduction ────────────────────────────────────────────────

    def _accumulate(self, covering, targets: list, hash_part: Any, from_layout: bool = False) -> FieldList:
        """Cover every target's window, fetch the fields, sum them, and check the result.

        Parameters
        ----------
        covering
            Resolves a window into the subwindows covering it.
        targets
            ``(valid_date, basetime)`` rows; *basetime* is ``None`` on the
            validity-date path.
        hash_part
            What identifies the source-data description in the subsource cache key.
        from_layout
            Whether the subsource is the run the trajectory layout imposes, so the
            covering is anchored on each row's basetime and the argument is a
            ``ForecastIntervals``.
        """
        windows = {}
        for vt, bt in targets:
            windows[(vt, bt)] = cut(covering, vt - self.period, vt, basetime=bt if from_layout else None)
            LOG.debug("  Covering of %s to %s:", vt - self.period, vt)
            for subwindow in windows[(vt, bt)]:
                LOG.debug("    %s", subwindow)

        source_object = create_source_object(self.context, self.source, (str(self.period), self.source, hash_part))
        input_fields = source_object(self.context, argument(targets, windows, self.period, forecast=from_layout))

        reducers, fields = reduce_ranges(
            input_fields,
            targets=targets,
            windows=windows,
            period=self.period,
            operation=OPERATIONS["sum"],
            group_by=self.group_by,
            field_to_interval=self._field_to_interval,
            source=self.source,
            stamp_basetime=True,
            declared="accum",
        )

        # MARS may answer a request with fields that do not match it exactly: runs at
        # 06Z and 18Z live in `scda` while the request says `oper`, so the windows
        # anchored on them get nothing at all. Drop those windows. "No field arrived"
        # is the test, not "no value yet": a from-zero window holding `a(0,12)` and
        # still waiting for `a(0,6)` has produced no value, and that half-window is an
        # error, reported by the completeness check.
        empty = {k for k, reducer in reducers.items() if reducer.fields_used == 0}
        for k in empty:
            LOG.warning("accumulate: no field reached the window for %s; dropping it", k)
            del reducers[k]

        check_complete("accumulate", self.period, reducers, fields, targets, discarded=empty)

        ds = apply_clip(FieldList.from_fields(fields), self.clip)
        LOG.debug("Created %d accumulated fields:", len(ds))
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
            return covering_from_description(description, period=self.period, source_name="accumulate")

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
        for d in dates:
            if not isinstance(d, datetime.datetime):
                raise TypeError("valid_date must be a datetime.datetime instance")
        return self._accumulate(self._searched_covering(), [(d, None) for d in dates], self._description_hash_part())

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
          a ``lookup-table``) — reconstructed by the same base-less covering
          *search* as :meth:`execute_valid_dates`; only the result is stamped
          as a forecast field at ``(basetime, step)``.
        """
        LOG.debug("💬 source for forecast accumulations: %s", self.source)
        description = self._resolved_from()
        targets = list(dates.items)

        if isinstance(description, FromTrajectories) and description.is_layout_grid:
            covering = ForecastCovering(period=self.period, accumulation=description.accumulation)
            return self._accumulate(covering, targets, description.accumulation, from_layout=True)

        return self._accumulate(self._searched_covering(), targets, self._description_hash_part())
