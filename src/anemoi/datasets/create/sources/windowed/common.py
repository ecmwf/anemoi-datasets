# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Helpers shared by every window source: subsource set-up, grouping, completeness."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from anemoi.utils.dates import frequency_to_string

LOG = logging.getLogger(__name__)


def prepare_source(name: str, source: Any, mars_type: str) -> str:
    """Validate the subsource config, apply MARS defaults, and return the subsource name."""
    if not (isinstance(source, dict) and len(source) == 1):
        raise ValueError(f"{name}: 'source' must have exactly one key, got {sorted(source)}")
    source_name, source_config = next(iter(source.items()))
    if source_name == "mars":
        if "type" not in source_config:
            source_config["type"] = mars_type
            LOG.warning(f"{name}: assuming 'type: {mars_type}' for the mars source as the recipe did not specify one")
        if "levtype" not in source_config:
            source_config["levtype"] = "sfc"
            LOG.warning(f"{name}: assuming 'levtype: sfc' for the mars source as the recipe did not specify one")
    return source_name


def create_source_object(context: Any, source: Any, key_parts: tuple) -> Any:
    """A cached subsource object, keyed by a content hash of *key_parts*."""
    h = hashlib.md5(json.dumps(key_parts, sort_keys=True, default=str).encode()).hexdigest()
    return context.create_source(source, "data_sources", h)


def group_key(field: Any, group_by: dict) -> tuple:
    """The metadata identifying the variable a field belongs to."""
    meta = field.get(collections=f"metadata.{group_by['namespace']}")
    return tuple(sorted((k, v) for k, v in meta.items() if k not in group_by["ignore"]))


def describe(reducers: dict, limit: int = 20) -> str:
    """Render reducers for an error message, least complete first."""
    ordered = sorted(reducers.values(), key=lambda r: (-r.outstanding, r.valid_date))
    lines = [f"  {r.__repr__(verbose=True)}" for r in ordered[:limit]]
    if len(ordered) > limit:
        lines.append(f"  ... and {len(ordered) - limit} more")
    return "\n".join(lines)


def check_complete(name: str, period, reducers: dict, fields: list, targets: list, discarded=()) -> None:
    """Raise unless every window came out whole.

    Two checks, because they catch different things. A reducer that exists but never
    filled is an incomplete window. A ``(target, variable)`` pair that never produced a
    reducer at all leaves nothing to be incomplete, so only comparing against the
    targets finds it. *discarded* reducer keys were dropped on purpose and are exempt.
    """
    if not reducers:
        raise ValueError(f"{name}: the source returned no usable field, cannot reduce anything")

    incomplete = {k: r for k, r in reducers.items() if not r.is_complete()}
    if incomplete:
        raise ValueError(
            f"{name}: {len(incomplete)} window(s) are missing source fields — a reduction over an "
            "incomplete window would silently bias the result and its statistics:\n"
            f"{describe(incomplete)}"
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
        raise ValueError(f"{name}: no source data at all for {len(missing)} (date, variable) combination(s):\n{detail}")

    # The bounds, not just the length: the window is end-anchored and half-open, and a
    # reader who assumes a centred one is wrong by half the period.
    LOG.info("%s: created %d field(s), each over (date - %s, date]", name, len(fields), frequency_to_string(period))
