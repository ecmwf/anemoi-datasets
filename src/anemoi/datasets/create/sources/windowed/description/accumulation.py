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

import datetime
import re

from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

ACCUMULATION_VALUES = (
    "'from-zero', a duration (e.g. '1h', the length each field holds), "
    "or 'from-zero-reset-every-<frequency>' (e.g. 'from-zero-reset-every-24h')"
)

_RESET_RE = re.compile(r"^from-zero-reset-every-(.+)$")


def _check_duration(value: datetime.timedelta, what: str) -> datetime.timedelta:
    """Validate a scheme duration: positive, and a whole number of minutes."""
    if value <= datetime.timedelta(0):
        raise ValueError(f"{what} must be positive, got {frequency_to_string(value)}")
    if value.total_seconds() % 60:
        raise ValueError(f"{what} must be a whole number of minutes, got {frequency_to_string(value)}")
    return value


def parse_accumulation(value: str) -> tuple[str, datetime.timedelta | None]:
    """Parse an ``accumulation:`` value.

    Parameters
    ----------
    value
        ``from-zero``, a duration (e.g. ``1h`` or ``10m`` — the fixed length
        each field holds, which used to be spelled ``from-previous-step``),
        or ``from-zero-reset-every-<frequency>``.

    Returns
    -------
    tuple
        ``(kind, length)`` where *kind* is ``"from-zero"``, ``"increment"``
        or ``"from-zero-reset"``.  *length* is the increment length (for
        ``"increment"``), the reset frequency (for ``"from-zero-reset"``),
        both as timedeltas, or ``None`` (for ``"from-zero"``).
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
        return "from-zero-reset", _check_duration(reset, f"Reset frequency in 'accumulation: {value}'")
    # A bare duration: the fixed accumulation length each field holds (the
    # former 'from-previous-step', now stated as the length itself).
    try:
        increment = frequency_to_timedelta(value)
    except Exception:
        raise ValueError(f"Invalid 'accumulation' value {value!r}; expected one of {ACCUMULATION_VALUES}")
    return "increment", _check_duration(increment, f"The duration in 'accumulation: {value}'")
