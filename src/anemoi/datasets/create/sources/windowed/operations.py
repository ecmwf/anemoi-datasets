# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The operations that reduce a window's parts to one value.

A window is resolved into parts -- subwindows for interval-valued source data, samples
for instant-valued -- each part yields one value and one weight, and an
:class:`Operation` folds them into the output field.

An operation never sees a sign and never sees an archived interval. Reconstructing a
subwindow from cumulative fields is a signed sum that happens *before* this, inside the
part, which is why ``max`` is no harder to express than ``sum``:

    reduce(reduce(reduce(None, v0, w0), v1, w1), v2, w2)

:attr:`Operation.differenceable` records the other half of that split: whether a part
carrying this operation's statistic may be rebuilt by differencing at all. It is a
statement about the *data*, not about the reduction.

NaNs propagate (``np.maximum``, not ``np.fmax``), as they do for a sum.
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


def _register_min_time_method() -> None:
    """Teach earthkit-data the ``min`` time-processing method.

    earthkit-data ships ``accum``, ``avg``, ``instant`` and ``max`` but no ``min``
    (still true in 1.2.3), so stamping ``proc.time_method="min"`` on a field raises
    ``Unsupported time method type: min``. anemoi-transform is already ready for it --
    ``_STEP_TYPE_FOR_CONVERSION`` maps ``min`` to the ``minimum`` statistical process --
    so the gap is only in the whitelist that validates the value.

    Without this, ``minimum`` would have to leave its output unstamped and the field
    would claim to be instantaneous. The registration is idempotent and disappears of
    its own accord once earthkit-data adds ``MIN``.
    """
    from earthkit.data.field.component import time_span

    if "min" not in time_span._TIME_METHODS:
        time_span._TIME_METHODS["min"] = time_span.TimeMethod("min")


_register_min_time_method()


#: The recipe block that produces each archived statistic, for error messages.
_BLOCK_FOR = {"accum": "'accumulate:'", "max": "'maximum:'", "min": "'minimum:'", "avg": "'average:'"}


class Operation(ABC):
    """Reduce the part values of a window to one value."""

    name: str

    #: The earthkit ``proc.time_method`` stamped on the output. The values match the
    #: GRIB step types anemoi-transform maps to a statistical process (``avg`` ->
    #: average, ``min`` -> minimum, ``max`` -> maximum, ``accum`` -> accumulation).
    time_method: str

    #: Whether a part carrying this statistic may be reconstructed by differencing
    #: archived fields. See :meth:`why_not_differenceable`.
    differenceable: bool

    @abstractmethod
    def reduce(self, reduced: NDArray | None, value: NDArray, weight: float) -> NDArray:
        """Reduce one part's value into the running window value.

        Parameters
        ----------
        reduced : numpy.ndarray or None
            The window value so far, or ``None`` for the first part.
        value : numpy.ndarray
            One part's values. The array belongs to the part that produced it and is
            released afterwards, so it may be consumed freely; it is never the shared
            array read off a field.
        weight : float
            The part's weight -- its length in seconds for a subwindow, ``1`` for a
            sample. Only length-weighted operations use it.

        Returns
        -------
        numpy.ndarray
            The updated window value.
        """

    def finalize(self, reduced: NDArray, total_weight: float) -> NDArray:
        """Turn the reduced value into the output, once the window is complete.

        Called exactly once, just before the field is built. The default returns
        *reduced* unchanged; :class:`Mean` divides by the total weight.

        Parameters
        ----------
        reduced : numpy.ndarray
            The value reduced over every part.
        total_weight : float
            The parts' weights, summed.

        Returns
        -------
        numpy.ndarray
            The values to write.
        """
        return reduced

    def reduces_archived(self, statistic: str | None) -> bool:
        """Whether this reduction may be applied to parts carrying *statistic*.

        Three cases are allowed and everything else is refused as not implemented:

        - the field does not say (``instant``, unknown) -- most archives cannot state
          it, so refusing here would reject nearly everything;
        - the statistic matches what this reduction produces (:attr:`time_method`), so
          the result is independent of how the window was partitioned;
        - the parts are **additive** and the reduction is an extremum -- "the wettest
          hour in the window". This one *is* partition-dependent, which is why
          ``over:`` exists to state the length.

        What is refused is a reduction over parts carrying a *different* statistic:
        the mean of block maxima, the sum of block maxima, the mean of block totals.
        Each is well defined only once the block length is stated, and producing it
        for a length the archive does not hold would mean combining stored values
        with their own operator -- which is a level of reconstruction this model does
        not have. See :meth:`why_not_archived`.
        """
        if statistic in (None, "instant", "unknown"):
            return True
        if statistic == self.time_method:
            return True
        return statistic == "accum" and self.name in ("max", "min")

    def why_not_archived(self, statistic: str) -> str:
        """Why this reduction may not be applied to parts carrying *statistic*."""
        block = _BLOCK_FOR.get(statistic, f"a {statistic!r} block")
        lines = [
            f"{self.name!r} over source data whose fields carry a {statistic!r} is not " "implemented.",
            f"The result would be the {self.name} of whatever blocks the archive happens to "
            "store, so its value would come from the archive's granularity rather than from "
            "the recipe -- and an archive whose granularity changes with lead time would "
            "give two different quantities inside one dataset.",
        ]
        if statistic == "accum" and self.name == "mean":
            lines.append(
                "For a mean *rate* over the window, accumulate and divide by the period; "
                "averaging the blocks gives the total divided by however many there were."
            )
        else:
            lines.append(
                f"Making it well defined needs both a way to state the block length and a way "
                f"to build blocks the archive does not hold, by combining stored values with "
                f"{statistic!r} -- and this model reconstructs a part only by a signed sum."
            )
        lines.append(f"Reduce this source data with {block}, which does not depend on the partition.")
        return " ".join(lines)

    def why_not_differenceable(self) -> str:
        """Why a part carrying this statistic cannot be rebuilt by differencing.

        Only meaningful when :attr:`differenceable` is false.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Sum(Operation):
    """Add the part values -- the operation ``accumulate`` performs.

    The only differenceable statistic here, and the reason ``accumulate`` works on
    archives that store values accumulated from the start of the forecast: an
    accumulation is additive, so ``a(6,12) = a(0,12) - a(0,6)``.
    """

    name = "sum"
    time_method = "accum"
    differenceable = True

    def reduce(self, reduced: NDArray | None, value: NDArray, weight: float) -> NDArray:
        if reduced is None:
            return value
        reduced += value
        return reduced


class _Extremum(Operation):
    """Shared implementation for ``max`` and ``min``."""

    differenceable = False

    def __init__(self, op) -> None:
        self._op = op

    def reduce(self, reduced: NDArray | None, value: NDArray, weight: float) -> NDArray:
        if reduced is None:
            return value
        return self._op(reduced, value, out=reduced)

    def why_not_differenceable(self) -> str:
        return (
            f"a {self.name} over part of a window is not recoverable by subtracting two "
            "cumulative extrema -- if the extreme value falls in the earlier part, both "
            "fields carry it and the difference is zero"
        )


class Max(_Extremum):
    """Largest part value over the window (e.g. maximum wind gust)."""

    name = "max"
    time_method = "max"

    def __init__(self) -> None:
        super().__init__(np.maximum)


class Min(_Extremum):
    """Smallest part value over the window."""

    name = "min"
    time_method = "min"

    def __init__(self) -> None:
        super().__init__(np.minimum)


class Mean(Operation):
    """Weighted average of the part values.

    Each part is a statistic over its own part of the window, so the window average is
    the weighted mean of the parts::

        mean = sum(weight_i * value_i) / sum(weight_i)

    Samples weigh 1 each, giving the plain arithmetic mean. Subwindows weigh their own
    length, which matters whenever the partition mixes lengths -- an archive that is
    hourly out to step 6 and 3-hourly beyond it would otherwise count a 3 h subwindow
    the same as a 1 h one.

    This averages *part statistics*, not snapshots of a continuous field; averaging
    instantaneous fields is what the ``average`` source does with samples.
    """

    name = "mean"
    time_method = "avg"

    # A mean is additive once multiplied by its length -- m(6,9) = (9*m(0,9) -
    # 6*m(0,6)) / 3 -- so unlike max and min this is not impossible, merely not
    # implemented: reconstruction is a plain signed sum and would have to weight each
    # contribution by its length and renormalise.
    differenceable = False

    def reduce(self, reduced: NDArray | None, value: NDArray, weight: float) -> NDArray:
        if weight <= 0:
            raise ValueError(f"{self.name} needs a positive part weight, got {weight}")
        value *= weight
        if reduced is None:
            return value
        reduced += value
        return reduced

    def finalize(self, reduced: NDArray, total_weight: float) -> NDArray:
        if total_weight <= 0:
            raise ValueError(f"cannot average over a zero-weight window ({total_weight})")
        reduced /= total_weight
        return reduced

    def why_not_differenceable(self) -> str:
        return (
            "a mean is reconstructible in principle -- m(6,9) = (9*m(0,9) - 6*m(0,6)) / 3 "
            "-- but only with each contribution weighted by its length, and reconstruction "
            "here is a plain signed sum; the weighted form is not implemented"
        )


_OPERATIONS: dict[str, type[Operation]] = {o.name: o for o in (Sum, Max, Min, Mean)}

#: ``avg`` is the GRIB spelling; accept it too.
_ALIASES = {"avg": "mean", "average": "mean", "maximum": "max", "minimum": "min"}


def operation_factory(operation: str | Operation | None) -> Operation:
    """Build an :class:`Operation` from a name.

    Parameters
    ----------
    operation : str or Operation or None
        ``"sum"`` (the default), ``"max"``, ``"min"`` or ``"mean"``; the recipe
        spellings ``maximum`` / ``minimum`` / ``average`` / ``avg``; an already-built
        operation; or ``None``.

    Returns
    -------
    Operation
        The operation.
    """
    if operation is None:
        return Sum()
    if isinstance(operation, Operation):
        return operation
    name = _ALIASES.get(operation, operation)
    if name not in _OPERATIONS:
        raise ValueError(
            f"Unknown operation {operation!r}; expected one of {sorted(_OPERATIONS)} or {sorted(_ALIASES)}"
        )
    return _OPERATIONS[name]()
