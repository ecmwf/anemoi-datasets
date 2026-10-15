# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The four reductions: sum, max, min and mean.

Each window is cut into pieces (subwindows or samples); each piece yields one array and
one weight, and an operation reduces them into the output values. What may be reduced
by what is not decided here -- see :mod:`.check`.

NaNs propagate (``np.maximum``, not ``np.fmax``), as they do for a sum.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from anemoi.transform.fields import Field
from numpy.typing import NDArray


def _register_min_time_method() -> None:
    """Teach earthkit-data the ``min`` time-processing method.

    earthkit-data ships ``accum``, ``avg``, ``instant`` and ``max`` but no ``min``
    (still true in 1.2.3), so stamping ``proc.time_method="min"`` on a field raises
    ``Unsupported time method type: min``. anemoi-transform already maps ``min`` to the
    ``minimum`` statistical process, so the gap is only in the whitelist that validates
    the value. Idempotent, and redundant once earthkit-data adds ``MIN``.
    """
    from earthkit.data.field.component import time_span

    if "min" not in time_span._TIME_METHODS:
        time_span._TIME_METHODS["min"] = time_span.TimeMethod("min")


_register_min_time_method()


class Operation(NamedTuple):
    """How the pieces of a window are reduced to one value.

    ``reduce(reduced, value, weight)`` combines one piece into the running value, which is
    ``None`` before the first piece; it may consume *value*. ``finalize(reduced,
    total_weight)`` runs once, when the window is complete.
    """

    name: str
    #: The earthkit ``proc.time_method`` stamped on the output.
    time_method: str
    reduce: Callable[[NDArray | None, NDArray, float], NDArray]
    finalize: Callable[[NDArray, float], NDArray] = lambda reduced, total_weight: reduced


def _sum(reduced, value, weight):
    if reduced is None:
        return value
    reduced += value
    return reduced


def _max(reduced, value, weight):
    return value if reduced is None else np.maximum(reduced, value, out=reduced)


def _min(reduced, value, weight):
    return value if reduced is None else np.minimum(reduced, value, out=reduced)


def _weighted_sum(reduced, value, weight):
    # A piece weighs its length (a sample weighs 1), so a partition mixing 1 h and
    # 3 h pieces averages to the mean over the window, not over the pieces.
    value *= weight
    return _sum(reduced, value, weight)


def _divide(reduced, total_weight):
    reduced /= total_weight
    return reduced


OPERATIONS = {
    "sum": Operation("sum", "accum", _sum),
    "max": Operation("max", "max", _max),
    "min": Operation("min", "min", _min),
    "mean": Operation("mean", "avg", _weighted_sum, _divide),
}

#: The recipe block for each operation.
BLOCKS = {"sum": "accumulate", "max": "maximum", "min": "minimum", "mean": "average"}


def stamp(
    values: NDArray,
    template: Field,
    operation: Operation,
    valid_date: datetime.datetime,
    period: datetime.timedelta,
    basetime: datetime.datetime | None,
) -> Field:
    """Build the output field for one reduced window.

    Without a basetime the base time is the start of the window and the step reaches
    the validity time, so the whole step is the window. For a trajectory row the base
    time is the model-run basetime, so trajectory loaders recover ``(basetime, step)``.
    """
    if basetime is None:
        basetime = valid_date - period
    return Field.from_numpy(
        values,
        template=template,
        **{
            "time.base_datetime": basetime,
            "time.step": valid_date - basetime,
            "proc.time_method": operation.time_method,
            "proc.time_value": period,
        },
    )
