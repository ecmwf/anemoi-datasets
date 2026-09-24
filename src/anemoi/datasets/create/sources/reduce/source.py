# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ``reduce`` source: arbitrary temporal reductions over a window.

``reduce`` builds one output field per validity date by folding the archived
fields that **tile** ``[valid_date - period, valid_date]`` with a single
operation — ``sum``, ``max``, ``min`` or ``mean``.

It is deliberately narrower than :mod:`~anemoi.datasets.create.sources.accumulate`.
``accumulate`` can *reconstruct* a window by subtracting archived fields
(``a(6,12) = +a(0,12) - a(0,6)``), which is what makes accumulated-from-start
archives usable. ``reduce`` never subtracts: it only ever applies the declared
operation to fields that already cover disjoint pieces of the window. That
restriction is what allows non-invertible operations such as ``max``, where the
maximum over ``[6,12]`` simply cannot be recovered from the maxima over
``[0,12]`` and ``[0,6]``, and ``mean``, which weights each field by the length
of the interval it covers and so needs those lengths to partition the window.

Practical consequence: a quantity stored accumulated from the start of the
forecast would have to be de-aggregated first, and this is currently out of scope. The
source detects it while building the covering — before any data is retrieved —
and fails with an explanatory error.

Everything else (covering search, grouping, metadata patching, GRIB output) is
shared with ``accumulate``.
"""

import datetime
import logging
from typing import Any

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.sources import source_registry

from ..accumulate.field_to_interval import FieldToInterval
from ..accumulate.operations import operation_factory
from ..accumulate.source import AccumulateSource

LOG = logging.getLogger(__name__)


@source_registry.register("reduce")
class ReduceSource(AccumulateSource):
    """Aggregate the fields tiling a window with one operation.

    Parameters
    ----------
    context
        Build context.
    source
        Single-key dict naming the backend and its request, e.g.
        ``{"mars": {...}}``.  One of ``mars``, ``fdb``, ``grib-index``.
    period
        Length of the window to reduce over (e.g. ``6h``).
    covering
        How the archive lays out its intervals; same shapes as ``accumulate``.
        Must resolve to a forward-only tiling of every window.
    operation
        ``sum`` (default), ``max``, ``min`` or ``mean`` (also spelled ``avg``).
    patch
        Metadata patches applied to returned fields, as for ``accumulate``.
    group_by
        Which metadata keys identify a field group, as for ``accumulate``.
    """

    # reduce aggregates archived fields directly; it never reconstructs a window
    # by subtraction.  This is the structural difference from `accumulate`.
    POSITIVE_ONLY = True

    SUPPORTED_SOURCES = ("mars", "fdb", "grib-index")

    def __init__(
        self,
        context: Any,
        source: Any,
        period: str | int | datetime.timedelta,
        covering=None,
        operation: str = "sum",
        patch: Any = None,
        group_by: dict | None = None,
    ) -> None:
        super().__init__(
            context,
            source=source,
            period=period,
            covering=covering,
            patch=patch,
            group_by=group_by,
        )
        self.operation = operation_factory(operation)

        # `accumulate` reads a field with startStep == endStep as covering [0, endStep];
        # for `reduce` that would let an instantaneous field masquerade as a window
        # statistic and be written out as, say, a 6h mean of a single snapshot.
        self._field_to_interval = FieldToInterval(patch, require_interval=True)

        if self._source_name not in self.SUPPORTED_SOURCES:
            raise ValueError(
                f"Source {self._source_name!r} is not supported by 'reduce'; "
                f"expected one of {list(self.SUPPORTED_SOURCES)}."
            )

    def execute_valid_dates(self, dates: ValidDates) -> Any:
        """Reduce each window ``[date - period, date]`` to a single field."""
        if self.covering is None:
            raise ValueError(
                "Argument 'covering' must be specified for the reduce source. See "
                "https://anemoi.readthedocs.io/projects/datasets/en/latest/building/sources/reduce.html"
            )
        return super().execute_valid_dates(dates)

    def execute_forecast_dates(self, dates: ForecastDates) -> Any:
        """Not supported: ``reduce`` has no trajectory branch yet."""
        raise NotImplementedError(
            "The 'reduce' source does not support trajectory recipes yet. Its covering is "
            "resolved from 'covering:', which the trajectory branch does not use."
        )

    def execute_intervals(self, dates: Intervals) -> Any:
        """Not supported: nesting a reduce inside another window source."""
        raise NotImplementedError(
            "The 'reduce' source cannot consume already-resolved Intervals "
            "(it owns its own covering). Do not nest it inside another accumulate/reduce block."
        )
