# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Which reductions may be applied to which archived statistics -- in one place.

Every window source calls :func:`check` once per variable, when the first field of
that variable arrives, because that is when the archived statistic is known. Nothing
else in the package decides validity.

The rows are the operation (named by the recipe block), the columns what each archived
field means over its own interval. ``instant`` -- which is also what a GRIB1 field that
does not say looks like -- is accepted everywhere, on trust.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from anemoi.utils.dates import frequency_to_string

from .operations import BLOCKS

#: Partition-invariant: the answer does not depend on how the window was cut.
OK = "ok"
#: The mean / extremum of the archive's own subwindows: only defined when every one is
#: held outright and all have the same length.
EQUAL_SUBWINDOWS = "equal"
#: An extremum of additive pieces ("the wettest hour"): needs at least two pieces per
#: window, all of the same length.
EXTREMUM_OF_TOTALS = "totals"

_CROSSED = "the {op} of stored {stat}s is an extremum of a different statistic; use '{same}:'"
_SUM_OF = "adding up stored {stat}s gives no physical quantity; use '{same}:'"

TABLE: dict[tuple[str, str], str] = {
    ("sum", "accum"): OK,
    ("sum", "max"): _SUM_OF,
    ("sum", "min"): _SUM_OF,
    ("sum", "avg"): _SUM_OF,
    ("max", "accum"): EXTREMUM_OF_TOTALS,
    ("max", "max"): OK,
    ("max", "min"): _CROSSED,
    ("max", "avg"): EQUAL_SUBWINDOWS,
    ("min", "accum"): EXTREMUM_OF_TOTALS,
    ("min", "max"): _CROSSED,
    ("min", "min"): OK,
    ("min", "avg"): EQUAL_SUBWINDOWS,
    ("mean", "accum"): (
        "averaging stored totals gives the window total divided by however many subwindows the "
        "archive holds; for a mean rate, use 'accumulate:' and divide by the period"
    ),
    ("mean", "max"): EQUAL_SUBWINDOWS,
    ("mean", "min"): EQUAL_SUBWINDOWS,
    ("mean", "avg"): OK,
}

#: Statistics for which subtracting two fields does not give the statistic over the
#: difference of their intervals.
NON_ADDITIVE = frozenset({"max", "min", "avg"})

_STAT_OPERATION = {"accum": "sum", "max": "max", "min": "min", "avg": "mean"}


@dataclass(frozen=True)
class Pieces:
    """How the windows were cut, summarised for :func:`check`.

    Parameters
    ----------
    lengths : frozenset of datetime.timedelta
        The distinct piece lengths over all windows.
    differenced : str or None
        One piece rebuilt from several archived fields, for the error message; ``None``
        if every piece is held outright.
    fewest : int
        The smallest number of pieces in any window.
    """

    lengths: frozenset
    differenced: str | None
    fewest: int

    @classmethod
    def of_subwindows(cls, windows: Iterable[list]) -> Pieces:
        """Summarise lists of :class:`~.subwindows.Subwindow`, one list per window."""
        windows = list(windows)
        differenced = next((s for w in windows for s in w if not s.is_direct), None)
        return cls(
            lengths=frozenset(s.end - s.start for w in windows for s in w),
            differenced=None if differenced is None else _describe(differenced),
            fewest=min((len(w) for w in windows), default=0),
        )

    @classmethod
    def of_samples(cls, frequency: datetime.timedelta, windows: Iterable[list]) -> Pieces:
        """Summarise sample times: each held outright, all one ``frequency`` long."""
        return cls(lengths=frozenset({frequency}), differenced=None, fewest=min((len(w) for w in windows), default=0))


def _describe(subwindow) -> str:
    signed = " ".join(f"{'+' if c.sign > 0 else '-'}{c}" for c in subwindow.contributions)
    return f"{subwindow} = {signed}"


def check(operation: str, statistic: str, pieces: Pieces) -> None:
    """Raise unless *operation* may reduce pieces of a field carrying *statistic*.

    Parameters
    ----------
    operation : str
        ``sum``, ``max``, ``min`` or ``mean``.
    statistic : str
        What each archived field means over its interval: ``accum``, ``max``, ``min``,
        ``avg``, or ``instant`` when the field does not say.
    pieces : Pieces
        How the windows were cut.

    Raises
    ------
    ValueError
        With the reason, and what to write instead.
    """
    if statistic not in _STAT_OPERATION:
        return

    block = f"'{BLOCKS[operation]}:'"
    verdict = TABLE[(operation, statistic)]
    if verdict not in (OK, EQUAL_SUBWINDOWS, EXTREMUM_OF_TOTALS):
        reason = verdict.format(op=operation, stat=statistic, same=BLOCKS[_STAT_OPERATION[statistic]])
        raise ValueError(f"{block} over fields that store a {statistic!r} is not supported: {reason}.")

    if statistic in NON_ADDITIVE and pieces.differenced:
        extra = ""
        if statistic == "avg":
            extra = (
                " (a mean from the start of the run would need weighting by length,"
                " m(6,9) = (9*m(0,9) - 6*m(0,6)) / 3, which is not implemented)"
            )
        raise ValueError(
            f"{block}: the fields store a {statistic!r}, which is not additive, but a piece of the "
            f"window would be rebuilt as a signed sum of several fields:\n    {pieces.differenced}\n"
            f"Subtracting or adding two {statistic!r} fields does not give the {statistic!r} over the "
            f"resulting interval{extra}. Check that 'from:' describes how this parameter is stored; "
            "'over:' only applies to additive fields."
        )

    lengths = ", ".join(frequency_to_string(x) for x in sorted(pieces.lengths))

    if verdict == EQUAL_SUBWINDOWS and len(pieces.lengths) > 1:
        raise ValueError(
            f"{block} over fields that store a {statistic!r} reduces the archive's own subwindows, which is "
            f"only defined when they all have the same length; here they are {lengths}."
        )

    if verdict == EXTREMUM_OF_TOTALS:
        if pieces.fewest < 2:
            raise ValueError(
                f"{block} over accumulations would reduce a single piece, so the result would be the "
                f"window's total labelled as a {operation}. Say how long each piece is with 'over:', "
                "e.g. 'over: 1h' for the largest hourly total in the window."
            )
        if len(pieces.lengths) > 1:
            raise ValueError(
                f"{block} over accumulations of different lengths ({lengths}) would compare totals "
                "over different durations. Say how long each piece is with 'over:'."
            )


def field_statistic(field: Any) -> str:
    """What statistic *field* carries over its interval: ``accum``, ``max``, ``min``,
    ``avg``, or ``instant`` when it does not say.

    ``proc.time_method`` is earthkit's reading of the message, right on GRIB2 and
    ``instant`` on GRIB1 whatever the field is; ``stepTypeForConversion`` comes from the
    *parameter* (``tp`` is an accumulation, ``10fg`` a maximum), which is where GRIB1
    keeps it.
    """
    for read in (lambda: field.get("proc.time_method"), lambda: field.metadata("stepTypeForConversion")):
        try:
            value = read()
        except Exception:  # pragma: no cover - backend-dependent
            continue
        if value is not None and str(value) not in ("instant", "unknown"):
            return str(value)
    return "instant"
