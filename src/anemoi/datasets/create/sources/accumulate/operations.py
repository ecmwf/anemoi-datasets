# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Operations combining the fields that cover a time window.

Used by the ``reduce`` source, which folds the archived fields tiling
``[valid_date - period, valid_date]`` into a single output field with one
operation: ``sum``, ``max``, ``min`` or ``mean``.

``accumulate`` uses :class:`Sum` implicitly and is the only caller that ever
combines *reversed* intervals: it can reconstruct a window by subtracting
archived fields (``a(6,12) = +a(0,12) - a(0,6)``). That is what
:attr:`Operation.invertible` records — whether an operation can consume a
reversed interval at all. ``max``, ``min`` and ``mean`` cannot: the maximum over
``[6,12]`` is not recoverable from the maxima over ``[0,12]`` and ``[0,6]``.
``reduce`` sidesteps the question entirely by requiring a forward-only tiling
for every operation.

NaN handling is uniform: NaNs propagate (``np.maximum``, not ``np.fmax``).
"""

from __future__ import annotations

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


class Operation(ABC):
    """Strategy combining the fields that cover a time window."""

    name: str
    invertible: bool
    grib_step_type: str

    @abstractmethod
    def combine(self, accumulated: NDArray | None, values: NDArray, sign: int, weight: float) -> NDArray:
        """Fold ``values`` into ``accumulated`` and return the new accumulated array.

        Parameters
        ----------
        accumulated
            The values gathered so far, or ``None`` for the first contribution.
        values
            The contributing field's values.  This array is **shared** between
            every accumulator the field contributes to, so implementations must
            neither alias nor mutate it.
        sign
            ``+1`` or ``-1``, from the covering interval.  Only an invertible
            operation is ever handed ``-1``.
        weight
            Length of the contributing interval, in seconds.  Only
            length-weighted operations (:class:`Mean`) use it.

        Returns
        -------
        NDArray
            The updated accumulated values.
        """

    def finalize(self, accumulated: NDArray, total_weight: float) -> NDArray:
        """Turn the folded values into the output field, once the window is complete.

        Called exactly once, just before writing.  The default returns
        ``accumulated`` unchanged; :class:`Mean` divides by the total weight.

        Parameters
        ----------
        accumulated
            The values gathered over the whole window.
        total_weight
            Sum of the contributing interval lengths, in seconds.  Under the
            tiling guarantee this equals the requested period.

        Returns
        -------
        NDArray
            The values to write.
        """
        return accumulated

    def output_edition(self, template: Any) -> int | None:
        """GRIB edition to force on the output.

        Called once per output field, just before writing.  Lets an operation
        escape an edition that cannot express it.  The default keeps the
        template's own edition.

        Parameters
        ----------
        template
            Field used as the GRIB template.

        Returns
        -------
        int | None
            The edition to write, or ``None`` for "unchanged".
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Sum(Operation):
    """Add the covering fields — the operation ``accumulate`` uses."""

    name = "sum"
    invertible = True
    grib_step_type = "accum"

    def combine(self, accumulated: NDArray | None, values: NDArray, sign: int, weight: float) -> NDArray:
        # `sign * values` allocates, so the result never aliases `values`.
        contribution = sign * values
        if accumulated is None:
            return contribution
        accumulated += contribution
        return accumulated


class _Extremum(Operation):
    """Shared implementation for the non-invertible ``max``/``min`` operations."""

    invertible = False

    def __init__(self, op: Any) -> None:
        self._op = op
        self._promotion_warned = False

    def combine(self, accumulated: NDArray | None, values: NDArray, sign: int, weight: float) -> NDArray:
        assert sign > 0, f"{self.name} cannot consume a reversed interval (sign={sign})"
        if accumulated is None:
            # `values` is shared between accumulators and we mutate in place below.
            return values.copy()
        return self._op(accumulated, values, out=accumulated)

    def output_edition(self, template: Any) -> int | None:
        # GRIB1 cannot state "maximum" or "minimum" in the product definition. Table 5
        # (time range indicator) has codes for average (3) and accumulation (4) but none
        # for extrema, so eccodes writes both as timeRangeIndicator=2 -- "valid over the
        # interval", statistic unspecified -- and reads that back as stepType=max
        # whatever was written.
        #
        # In GRIB1 the statistic is carried by the *parameter* instead: 10fg is a
        # maximum, mn2t6 a minimum, tp an accumulation. eccodes exposes that as the
        # param-derived key `stepTypeForConversion`, and create/gridded/result.py falls
        # back on it when the message says nothing. So an extremum survives edition 1
        # only when the parameter declares that same statistic.
        #
        # When it does not, writing edition 1 would either record the wrong statistic
        # (min on 10fg is read back as a maximum) or record none at all (max on tp
        # resolves to nothing and fails later in the build).
        # Promote to edition 2, where typeOfStatisticalProcessing states it in the message.
        # The MARS local definition # (class/stream/expver/type) survives the conversion;
        # the parameter is renamed by eccodes to the one that genuinely matches, e.g. 10fg -> min_i10fg.
        if template.metadata("edition") != 1:
            return None

        declared = template.metadata("stepTypeForConversion", default=None)
        if declared == self.grib_step_type:
            return None

        if not self._promotion_warned:
            self._promotion_warned = True
            LOG.warning(
                "Writing operation %r as GRIB2: edition 1 cannot encode it for this parameter "
                "(it declares stepTypeForConversion=%r). The output parameter will be renamed by "
                "eccodes to the one matching the statistic, so a 'rename:' keyed on the original "
                "name will not match.",
                self.name,
                declared,
            )
        return 2


class Max(_Extremum):
    """Largest value over the window (e.g. maximum wind gust)."""

    name = "max"
    grib_step_type = "max"

    def __init__(self) -> None:
        super().__init__(np.maximum)


class Min(_Extremum):
    """Smallest value over the window."""

    name = "min"
    grib_step_type = "min"

    def __init__(self) -> None:
        super().__init__(np.minimum)


class Mean(Operation):
    """Time-weighted average over the window.

    Each contributing field is a statistic over its own sub-interval, so the
    window average is the length-weighted mean of the pieces::

        mean = sum(length_i * values_i) / sum(length_i)

    Weighting matters whenever the covering mixes interval lengths (e.g. an
    archive that is hourly out to step 6 and 3-hourly beyond it); with a uniform
    covering it reduces to the plain arithmetic mean.  ``reduce`` guarantees the
    intervals tile the window exactly, so the weights sum to the requested
    period and there is no double counting.

    Note this averages *interval statistics*, not snapshots: it is not a way to
    average instantaneous fields, which tile nothing.
    """

    name = "mean"
    invertible = False
    grib_step_type = "avg"

    def combine(self, accumulated: NDArray | None, values: NDArray, sign: int, weight: float) -> NDArray:
        assert sign > 0, f"{self.name} cannot consume a reversed interval (sign={sign})"
        assert weight > 0, f"{self.name} needs a positive interval length, got {weight}"
        # `weight * values` allocates, so the result never aliases `values`.
        contribution = weight * values
        if accumulated is None:
            return contribution
        accumulated += contribution
        return accumulated

    def finalize(self, accumulated: NDArray, total_weight: float) -> NDArray:
        assert total_weight > 0, f"cannot average over a zero-length window ({total_weight})"
        accumulated /= total_weight
        return accumulated


_OPERATIONS: dict[str, type[Operation]] = {o.name: o for o in (Sum, Max, Min, Mean)}

# `avg` is the GRIB spelling; accept it so recipes can use either.
_ALIASES = {"avg": "mean", "average": "mean"}


def operation_factory(operation: str | Operation | None) -> Operation:
    """Build an :class:`Operation` from a recipe ``operation:`` value.

    Parameters
    ----------
    operation
        One of ``"sum"`` (default), ``"max"``, ``"min"``, ``"mean"`` (also
        spelled ``"avg"``/``"average"``); or an already-built ``Operation``; or
        ``None`` for the default.

    Returns
    -------
    Operation
        The operation strategy.
    """
    if operation is None:
        return Sum()
    if isinstance(operation, Operation):
        return operation
    operation = _ALIASES.get(operation, operation)
    if operation not in _OPERATIONS:
        raise ValueError(
            f"Unknown operation {operation!r}; expected one of " f"{sorted(_OPERATIONS)} (or {sorted(_ALIASES)})"
        )
    return _OPERATIONS[operation]()
