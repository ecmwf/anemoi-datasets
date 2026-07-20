# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ``from.accumulation:`` grammar (how the source data accumulates)."""

from __future__ import annotations

import re

from anemoi.utils.dates import frequency_to_timedelta

ACCUMULATION_VALUES = (
    "'from-zero', a duration (e.g. '1h', the length each field holds), "
    "or 'from-zero-reset-every-<frequency>' (e.g. 'from-zero-reset-every-24h')"
)

_RESET_RE = re.compile(r"^from-zero-reset-every-(.+)$")


def parse_accumulation(value: str) -> tuple[str, int | None]:
    """Parse an ``accumulation:`` value.

    Parameters
    ----------
    value
        ``from-zero``, a duration (e.g. ``1h`` — the fixed length each
        field holds, which used to be spelled ``from-previous-step``), or
        ``from-zero-reset-every-<frequency>``.

    Returns
    -------
    tuple
        ``(kind, hours)`` where *kind* is ``"from-zero"``, ``"increment"``
        or ``"from-zero-reset"``.  *hours* is the increment length (for
        ``"increment"``), the reset frequency (for ``"from-zero-reset"``),
        both in whole hours, or ``None`` (for ``"from-zero"``).
    """
    if not isinstance(value, str):
        raise ValueError(f"Invalid 'accumulation' value {value!r}; expected one of {ACCUMULATION_VALUES}")
    if value == "from-zero":
        return "from-zero", None
    m = _RESET_RE.match(value)
    if m:
        try:
            reset = frequency_to_timedelta(m.group(1))
        except Exception as e:
            raise ValueError(f"Invalid reset frequency in 'accumulation: {value}': {m.group(1)!r}") from e
        hours = reset.total_seconds() / 3600
        if not (hours.is_integer() and hours > 0):
            raise ValueError(f"Reset frequency in 'accumulation: {value}' must be a positive whole number of hours")
        return "from-zero-reset", int(hours)
    # A bare duration: the fixed accumulation length each field holds (the
    # former 'from-previous-step', now stated as the length itself).
    try:
        increment = frequency_to_timedelta(value)
    except Exception:
        raise ValueError(f"Invalid 'accumulation' value {value!r}; expected one of {ACCUMULATION_VALUES}")
    hours = increment.total_seconds() / 3600
    if not (hours.is_integer() and hours > 0):
        raise ValueError(f"Invalid 'accumulation' value {value!r}; a duration must be a positive whole number of hours")
    return "increment", int(hours)
