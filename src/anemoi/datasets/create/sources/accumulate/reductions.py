# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""How the fields covering an accumulation window are combined.

The ``accumulate`` source builds one output field per window
``[valid_date - period, valid_date]`` out of the archived fields covering it.
The *reduction* says how those fields are combined:

- ``sum`` — the classic accumulation (precipitation, radiation).  It is
  **invertible**: a window can be built by subtracting archived fields, e.g.
  ``a(6,12) = +a(0,12) - a(0,6)``.  That is what lets the covering search use
  negative intervals, and what makes ``accumulation: from-zero`` work.

- ``max`` / ``min`` — windowed extrema, e.g. a wind gust field holding "the
  largest gust since the last output".  These are **not** invertible: the
  maximum over ``[6,12]`` cannot be recovered from the maxima over ``[0,12]``
  and ``[0,6]``.  A non-invertible reduction is therefore only defined when the
  covering is an all-positive tiling of the window — in practice, when the
  archive stores per-step values (``from-previous-step``).  The covering layer
  enforces this; see :func:`..covering.validate_tiling`.

NaN handling follows ``sum``: NaNs propagate (``np.maximum``, not ``np.fmax``).
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Reduction(ABC):
    """Strategy combining the fields that cover an accumulation window."""

    name: str
    invertible: bool
    grib_step_type: str

    @abstractmethod
    def combine(self, accumulated: NDArray | None, values: NDArray, sign: int) -> NDArray:
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
            reduction is ever handed ``-1``.

        Returns
        -------
        NDArray
            The updated accumulated values.
        """

    def check_template(self, template: Any) -> None:
        """Raise if this reduction cannot be safely encoded on ``template``.

        Called once per output field, just before writing.  The default
        implementation accepts everything.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Sum(Reduction):
    """Additive accumulation — the historical behaviour of this source."""

    name = "sum"
    invertible = True
    grib_step_type = "accum"

    def combine(self, accumulated: NDArray | None, values: NDArray, sign: int) -> NDArray:
        # `sign * values` allocates, so the result never aliases `values`.
        contribution = sign * values
        if accumulated is None:
            return contribution
        accumulated += contribution
        return accumulated


class _Extremum(Reduction):
    """Shared implementation for the non-invertible ``max``/``min`` reductions."""

    invertible = False

    def __init__(self, op: Any) -> None:
        self._op = op

    def combine(self, accumulated: NDArray | None, values: NDArray, sign: int) -> NDArray:
        assert sign > 0, f"{self.name} cannot consume a reversed interval (sign={sign})"
        if accumulated is None:
            # `values` is shared between accumulators and we mutate in place below.
            return values.copy()
        return self._op(accumulated, values, out=accumulated)


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

    def check_template(self, template: Any) -> None:
        # GRIB1 has no way to say "minimum".  eccodes encodes both min and max as
        # timeRangeIndicator=2, so the message reads back as stepType=max and
        # create/gridded/result.py records it as `process=maximum`.  Verified with
        # eccodes 2.47.  Fail loudly rather than write a mislabelled field.
        if template.metadata("edition") == 1:
            raise ValueError(
                "reduction 'min' cannot be encoded in GRIB edition 1: it shares "
                "timeRangeIndicator=2 with 'max' and would be read back as a maximum. "
                "Use a source that delivers GRIB2, or reduction 'max'."
            )


_REDUCTIONS: dict[str, type[Reduction]] = {r.name: r for r in (Sum, Max, Min)}


def reduction_factory(reduction: str | Reduction | None) -> Reduction:
    """Build a :class:`Reduction` from a recipe ``reduction:`` value.

    Parameters
    ----------
    reduction
        One of ``"sum"`` (default), ``"max"``, ``"min"``; or an already-built
        ``Reduction``; or ``None`` for the default.

    Returns
    -------
    Reduction
        The reduction strategy.
    """
    if reduction is None:
        return Sum()
    if isinstance(reduction, Reduction):
        return reduction
    if reduction not in _REDUCTIONS:
        raise ValueError(f"Unknown reduction {reduction!r}; expected one of {sorted(_REDUCTIONS)}")
    return _REDUCTIONS[reduction]()
