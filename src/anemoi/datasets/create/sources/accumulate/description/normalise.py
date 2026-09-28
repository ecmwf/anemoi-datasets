# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Fold every accepted spelling of the source-data description into ``from:``.

:func:`normalise_from` is the *one* implementation shared by recipe-time
validation (:class:`~.schema.AccumulateSchema`) and the runtime source, so
the two cannot drift apart.
"""

from __future__ import annotations

import datetime
import warnings
from typing import Any

from anemoi.utils.dates import frequency_to_string

from .union import _FROM_ADAPTER
from .union import FromBare
from .union import FromLookupTable
from .union import FromTrajectories

MIGRATE_HINT = "run 'anemoi-datasets recipe --migrate <recipe>' to rewrite the recipe"


def normalise_from(
    *,
    from_: Any = None,
    accumulation: str | None = None,
    covering: Any = None,
    availability: Any = None,
    warn: bool = True,
) -> tuple[Any, Any]:
    """Fold every accepted spelling of the source-data description into ``from:``.

    This is the single implementation of the description rules, shared by
    :class:`AccumulateSchema` (recipe time) and ``AccumulateSource``
    (run time) so the two cannot drift apart.

    Parameters
    ----------
    from_
        The ``from:`` block: a mapping or a validated union member (``None``
        when omitted).
    accumulation
        The deprecated block-level scheme key, folded into a bare
        ``from: {accumulation: ...}``.
    covering, availability
        The pre-redesign spellings, which keep running through the legacy
        covering machinery and are returned untouched.
    warn
        Emit ``DeprecationWarning`` for deprecated spellings. Recipe-time
        callers keep the default; the runtime source passes ``False`` so a
        block does not warn a second time when the validated recipe is
        re-normalised at build time.

    Returns
    -------
    tuple
        ``(from_, covering)`` — the validated ``from:`` union member, or
        ``None`` (``from:`` omitted: the source data is recognised from the
        source at build time, unless a legacy *covering* is also returned),
        and the legacy covering payload.
    """
    # -- deprecated covering/availability ------------------------------------
    if availability is not None:
        if covering is not None:
            raise ValueError("accumulate: cannot specify both 'covering' and its deprecated alias 'availability'")
        covering = {"auto": availability}

    if covering is not None and warn:
        warnings.warn(
            f"'covering:'/'availability:' are deprecated; describe the source data with 'from:' ({MIGRATE_HINT}).",
            DeprecationWarning,
            stacklevel=3,
        )

    # -- exactly one description --------------------------------------------
    given = [
        name
        for name, value in (("accumulation", accumulation), ("covering", covering), ("from", from_))
        if value is not None
    ]
    if len(given) > 1:
        raise ValueError(f"accumulate: only one source-data description is allowed, got {sorted(given)}")

    if from_ is not None:
        return _validate_from(from_), None

    if covering is not None:
        return None, covering

    if accumulation is not None:
        if warn:
            warnings.warn(
                "block-level 'accumulation:' is deprecated; state it inside 'from:' "
                f"as 'from: {{accumulation: ...}}' instead ({MIGRATE_HINT}).",
                DeprecationWarning,
                stacklevel=3,
            )
        return _validate_from({"accumulation": accumulation}), None

    # `from:` is optional: with no description and no legacy covering, the
    # source data is recognised from the (well-known MARS) source at build
    # time. That "recognise from the source" default is `from_ = None` with no
    # covering — there is no `auto` value. Only well-known MARS archives are
    # recognised, so anything else still fails loudly at build time (with a
    # message naming the supported classes).
    return None, None


def check_valid_time_source(from_: Any, *, period: datetime.timedelta) -> None:
    """Check a bare ``from:`` used as base-less, validity-time-indexed source data.

    Only a :class:`FromBare` is checked, and only the rules intrinsic to the
    *description*: the accumulation must be a fixed duration and it must divide
    the requested ``period`` (the windows are summed).  A bare
    ``from:`` means base-less, validity-time-indexed source data in *any* output
    layout (in a trajectory recipe the field valid at ``base_date + step`` is
    relabelled onto that row; elsewhere it is summed by validity time), so these
    rules apply everywhere.

    Whether the *source* can actually serve base-less (``base=None``) intervals
    is **not** decided here: the mode is the caller's — ``accumulate`` builds the
    argument type (base-less :class:`Intervals`) from ``from:`` and hands it to
    the source, and a base-anchored source (mars/fdb/…) rejects it at that point
    (``Intervals.adjust_request`` needs a basetime).  It is not a property to
    declare per source.

    Parameters
    ----------
    from_
        A validated ``from:`` description (only ``FromBare`` is acted on).
    period
        The requested output accumulation.
    """
    if not isinstance(from_, FromBare):
        return

    duration = from_.duration
    if duration is None:
        raise ValueError(
            "accumulate: a bare 'from:' describes base-less source data indexed by validity time, "
            "so 'accumulation' must be a fixed duration (e.g. '3h'); "
            f"{from_.accumulation!r} ('from-zero'/'from-zero-reset') is a run scheme — declare the "
            "run grid too, with explicit 'base_dates' + 'steps' (or 'base_dates: from-layout, "
            "steps: from-layout' in a trajectory recipe)"
        )

    if period % duration != datetime.timedelta(0):
        raise ValueError(
            f"accumulate: the source data's 'accumulation' ({frequency_to_string(duration)}) "
            f"must divide the requested 'period' ({frequency_to_string(period)})"
        )


def _validate_from(value: Any) -> Any:
    """Validate a ``from:`` payload into a description (only called for a non-``None`` ``from:``)."""
    if isinstance(value, (FromTrajectories, FromBare, FromLookupTable)):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"accumulate: 'from:' must be a mapping, got {value!r}")

    return _FROM_ADAPTER.validate_python(value)
