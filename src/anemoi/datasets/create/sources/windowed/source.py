# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Base class for sources that build one field per date by reducing a time window.

``accumulate`` and the time reductions (``average``, ``minimum``, ``maximum``) do the
same work: resolve each window into parts, fetch the fields covering them, group them
by variable, reduce each window to one value, and stamp the result. They differ only in
what a part *is* and how an arriving field is matched to one, which is what
:class:`~.plan.WindowPlan` carries.

Subclasses provide the recipe surface, a plan, and what identifies their configuration
in the subsource cache key.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from typing import Any

from anemoi.transform import FieldList
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

from anemoi.datasets.create.source import Source

from .groupby import patch_groupby_keys
from .plan import Target
from .plan import WindowPlan
from .reducer import describe

LOG = logging.getLogger(__name__)


class WindowSourceBase(Source):
    """One output field per date, reduced from the window ending at it."""

    #: The registered recipe key, for error messages.
    name: str = "window"

    def __init__(
        self,
        context: Any,
        source: Any,
        period: str | int | datetime.timedelta,
        group_by: dict | None = None,
    ) -> None:
        super().__init__(context)
        self.source = source
        self.period = frequency_to_timedelta(period)
        self.group_by = patch_groupby_keys(group_by, source_name=self.name)
        self._source_name = self._prepare_source()

    # ── subclass contract ────────────────────────────────────────────

    def _hash_parts(self) -> tuple:
        """What distinguishes this configuration, for the subsource cache key."""
        return ()

    def _mars_type_default(self) -> str:
        """MARS ``type`` to assume when the recipe does not say.

        A forecast: interval-valued data is archived per step. The reductions override
        this, because base-less instantaneous fields are indexed by validity time.
        """
        return "fc"

    def _discard_unusable(self, reducers: dict) -> set:
        """Reducer keys to drop rather than report as a failure.

        The default is to drop nothing: a window that no field reached is an error,
        because a reduction over part of a window is silently wrong rather than
        merely incomplete. ``accumulate`` overrides this -- see its own docstring for
        the one case where an empty window is expected.

        Called before the completeness checks; the keys returned are also exempt from
        the missing-``(target, variable)`` check, since they were deliberate.
        """
        return set()

    # ── shared helpers ───────────────────────────────────────────────

    def _prepare_source(self) -> str:
        """Validate the subsource config and apply MARS defaults."""
        source = self.source
        if not (isinstance(source, dict) and len(source) == 1):
            raise ValueError(f"{self.name}: 'source' must have exactly one key, got {sorted(source)}")
        source_name, source_config = next(iter(source.items()))
        if source_name == "mars":
            if "type" not in source_config:
                source_config["type"] = self._mars_type_default()
                LOG.warning(
                    "%s: assuming 'type: %s' for the mars source as the recipe did not specify one",
                    self.name,
                    source_config["type"],
                )
            if "levtype" not in source_config:
                source_config["levtype"] = "sfc"
                LOG.warning(
                    "%s: assuming 'levtype: sfc' for the mars source as the recipe did not specify one", self.name
                )
        return source_name

    def _create_source_object(self, *extra_hash_parts: Any):
        """Create a cached subsource object keyed by a content hash."""
        key_parts = (self.name, str(self.period), self.source, *self._hash_parts(), *extra_hash_parts)
        h = hashlib.md5(json.dumps(key_parts, sort_keys=True, default=str).encode()).hexdigest()
        return self.context.create_source(self.source, "data_sources", h)

    def _group_key(self, field: Any) -> tuple:
        """The metadata identifying the variable a field belongs to."""
        meta = field.get(collections=f"metadata.{self.group_by['namespace']}")
        key = {k: v for k, v in meta.items() if k not in self.group_by["ignore"]}
        return tuple(sorted(key.items()))

    # ── the shared loop ──────────────────────────────────────────────

    def _reduce_fields(
        self,
        plan: WindowPlan,
        source_object: Any,
        argument: Any,
        targets: list[Target],
        parts: dict[Target, list],
    ) -> tuple[dict, list]:
        """Fetch the fields and reduce each window down to one field per variable.

        Parameters
        ----------
        plan
            Resolves parts and matches fields to them.
        source_object
            Subsource factory, called as ``source_object(context, argument)``.
        argument
            What to ask the subsource for.
        targets
            The ``(valid_date, basetime)`` rows to produce.
        parts
            Each target's window, as returned by ``plan.parts_for``.

        Returns
        -------
        tuple
            ``(reducers, fields)``.
        """
        # Execute the subsource once: the same FieldList feeds the loop and, on
        # failure, whatever diagnostics the plan wants to print.
        input_fields = source_object(self.context, argument)

        reducers: dict = {}
        fields: list = []
        plan.begin(input_fields, reducers)

        for field in input_fields:
            identity = plan.identify(field)
            # Read once, not once per target: this field is offered to every window
            # that might want it.
            info = plan.field_info(field)
            key = self._group_key(field)
            values = field.values

            used_by: list = []
            for target in plan.candidates(identity, targets):
                reducer_key = (*target, key)
                if reducer_key not in reducers:
                    reducers[reducer_key] = plan.new_reducer(target, key, parts[target])

                reducer = reducers[reducer_key]
                if plan.offer(reducer, values, identity, info):
                    used_by.append((target, reducer))
                    if reducer.is_complete():
                        fields.append(reducer.as_field(template=field))

            plan.note(field, identity, used_by)
            if not used_by:
                plan.unused_field(field, identity)

        return reducers, fields

    def _as_fieldlist(self, fields: list) -> FieldList:
        """Wrap the reduced fields; a subclass may post-process them."""
        return FieldList.from_fields(fields)

    def _run(self, plan: WindowPlan, targets: list[Target], *hash_parts: Any) -> FieldList:
        """Resolve the windows, fetch the fields, reduce them and check the result."""
        parts = plan.parts_for(targets)
        source_object = self._create_source_object(*hash_parts)
        reducers, fields = self._reduce_fields(plan, source_object, plan.argument(targets, parts), targets, parts)
        return self._finalise(reducers, fields, targets)

    def _finalise(self, reducers: dict, fields: list, targets: list[Target]) -> FieldList:
        """Check every window came out whole, and return the reduced fields.

        Two checks, because they catch different things. A reducer that exists but
        never filled is an incomplete window. A ``(target, variable)`` pair that never
        produced a reducer at all leaves nothing to be incomplete, so only comparing
        against the targets finds it.
        """
        discarded = self._discard_unusable(reducers)

        if not reducers:
            raise ValueError(f"{self.name}: the source returned no usable field, cannot reduce anything")

        incomplete = {k: r for k, r in reducers.items() if not r.is_complete()}
        if incomplete:
            raise ValueError(
                f"{self.name}: {len(incomplete)} window(s) are missing source fields — a "
                "reduction over an incomplete window would silently bias the result and its "
                f"statistics:\n{describe(incomplete)}"
            )

        keys = {key for *_, key in reducers}
        missing = [
            (t, key) for t in targets for key in sorted(keys) if (*t, key) not in reducers and (*t, key) not in discarded
        ]
        if missing:
            detail = "\n".join(
                f"  {vdate}{f' (basetime {basetime})' if basetime is not None else ''}: {dict(key)}"
                for (vdate, basetime), key in missing[:20]
            )
            raise ValueError(
                f"{self.name}: no source data at all for {len(missing)} (date, variable) combination(s):\n{detail}"
            )

        # The bounds, not just the length: the window is end-anchored and half-open, and a
        # reader who assumes a centred one is wrong by half the period in a way that looks
        # entirely plausible. This line is in every build log, so it is where the
        # convention costs nothing to state.
        LOG.info(
            "%s: created %d field(s), each over (date - %s, date]",
            self.name,
            len(fields),
            frequency_to_string(self.period),
        )
        return self._as_fieldlist(fields)
