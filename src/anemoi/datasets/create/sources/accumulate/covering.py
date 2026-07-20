# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Covering layer for the accumulate source.

The :class:`Covering` ABC produces a *covering* of an accumulation
window ``[start, end]`` — i.e. a list of ``SignedInterval`` objects
whose signed sum equals the window. It is intentionally separate from
``Availability`` (the ``IntervalGenerator`` family in
``interval_generators.py``) which only describes *what the source data
contains*.

Today there are three concrete strategies:

- :class:`AutoCovering` — search-based; wraps the existing Dijkstra
  over an ``IntervalGenerator``. Used by the validity-date
  path.
- :class:`ForecastCovering` — basetime-imposed; emits the trivial 1-
  or 2-interval decomposition for trajectory accumulations.
- :class:`ValidTimeCovering` — base-less; tiles the window directly
  for a validity-time-indexed increment source.

The ``forecast`` recipe discriminator is intentionally **not**
exposed: the trajectory branch is selected implicitly by the upstream
argument type (``ForecastDates``), not by the recipe.
"""

from __future__ import annotations

import datetime
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable

from anemoi.datasets.create.intervals import SignedInterval

from .interval_generators import IntervalGenerator
from .interval_generators import interval_generator_factory


def check_covering(
    covering: Iterable[SignedInterval],
    start: datetime.datetime,
    end: datetime.datetime,
) -> list[SignedInterval]:
    """Assert a covering's signed lengths add up to ``[start, end]``.

    The whole contract of this layer is that the accumulator can sum the
    covering's fields — each multiplied by its interval's sign — and get
    the accumulation over ``[start, end]``.  Nothing downstream re-checks
    it: ``Accumulator.is_complete()`` only verifies that every declared
    interval turned up, so a covering that does not add up would silently
    produce a wrong field.  Every :meth:`Covering.cover` implementation
    passes its result through here.

    Parameters
    ----------
    covering
        The signed intervals produced for the window.
    start, end
        The requested accumulation window.

    Returns
    -------
    list
        The covering, unchanged.
    """
    covering = list(covering)
    total = sum(i.length for i in covering)
    wanted = (end - start).total_seconds()
    if total != wanted:
        detail = "\n".join(f"    {'+' if i.length >= 0 else '-'} {i}" for i in covering)
        raise ValueError(
            f"Covering of {start} → {end} does not add up: its signed lengths total "
            f"{total / 3600:g}h, expected {wanted / 3600:g}h. The intervals were:\n{detail}\n"
            "Summing these would produce a wrong accumulation, so the build is stopped."
        )
    return covering


class Covering(ABC):
    """Strategy producing a covering of an accumulation window.

    Subclasses implement :meth:`cover` to return the list of
    ``SignedInterval`` objects whose signed sum equals
    ``[start, end]`` — enforced by :func:`check_covering`.
    """

    @abstractmethod
    def cover(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        *,
        basetime: datetime.datetime | None = None,
    ) -> Iterable[SignedInterval]:
        """Return a covering of ``[start, end]``.

        Parameters
        ----------
        start
            Start of the accumulation window.
        end
            End of the accumulation window.
        basetime
            Optional externally-imposed model-run time. ``AutoCovering``
            ignores it; ``ForecastCovering`` requires it.

        Returns
        -------
        Iterable[SignedInterval]
            The signed intervals covering ``[start, end]``.
        """


class AutoCovering(Covering):
    """Search-based covering over an :class:`IntervalGenerator`.

    Wraps the existing ``IntervalGenerator.covering_intervals`` search.
    ``basetime`` is not honoured (passing a non-``None`` basetime raises
    ``NotImplementedError``).
    """

    def __init__(self, availability: IntervalGenerator) -> None:
        self.availability = availability

    def cover(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        *,
        basetime: datetime.datetime | None = None,
    ) -> Iterable[SignedInterval]:
        if basetime is not None:
            raise NotImplementedError(
                "AutoCovering does not honour an externally-imposed basetime; "
                "use ForecastCovering for the trajectory case."
            )
        return check_covering(self.availability.covering_intervals(start, end), start, end)


class ForecastCovering(Covering):
    """Covering for trajectory accumulations.

    The basetime is dictated by the caller (e.g. via ``ForecastDates``);
    no search over the source data is performed. The covering is the trivial signed
    decomposition determined by the ``accumulation`` scheme:

    - ``"from-zero"``: the source data stores ``a(0, step)`` accumulations from
      the basetime. The window ``[basetime + sA, basetime + sE]`` is
      built as ``+a(0, sE) − a(0, sA)``.
    - a duration (e.g. ``"1h"``): the source data stores per-step increments
      ``a(step - length, step)``. The window is the single interval
      ``a(sA, sE)``.
    - ``"from-zero-reset-every-<freq>"``: the source data stores from-zero
      accumulations restarting every *freq* of lead time. Within one
      reset cycle the window is ``+a(r, sE) − a(r, sA)`` (r = last reset
      at or before sA); a window straddling reset boundaries adds one
      full-cycle interval per boundary crossed.

    Parameters
    ----------
    period
        Accumulation window length.
    accumulation
        One of the ``accumulation`` scheme values. There is no default —
        the caller must declare it explicitly.
    """

    def __init__(self, period: datetime.timedelta, accumulation: str) -> None:
        from .description import parse_accumulation

        # `_hours` is the scheme's hour parameter: the reset frequency for
        # from-zero-reset, the increment length for a duration, else None.
        self._kind, self._hours = parse_accumulation(accumulation)
        self.period = period
        self.accumulation = accumulation

    def cover(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        *,
        basetime: datetime.datetime | None = None,
    ) -> list[SignedInterval]:
        if basetime is None:
            raise ValueError("ForecastCovering.cover requires an explicit basetime.")

        delta_end = end - basetime
        delta_start = start - basetime
        step_end_h = delta_end.total_seconds() / 3600
        step_start_h = delta_start.total_seconds() / 3600

        if not (step_end_h.is_integer() and step_start_h.is_integer()):
            raise ValueError(
                "ForecastCovering requires integer-hour offsets between basetime "
                f"and the window endpoints; got start={step_start_h}, end={step_end_h}."
            )
        step_end_h = int(step_end_h)
        step_start_h = int(step_start_h)

        if step_start_h < 0:
            raise ValueError(
                f"Window {start}..{end} straddles basetime {basetime} "
                f"(step_start={step_start_h}h); not supported in v1."
            )
        if step_end_h <= step_start_h:
            raise ValueError(f"Window {start}..{end} has non-positive length relative to basetime {basetime}.")

        if self._kind == "from-zero":
            covering: list[SignedInterval] = []
            covering.append(
                SignedInterval(
                    start=basetime,
                    end=basetime + datetime.timedelta(hours=step_end_h),
                    base=basetime,
                )
            )
            if step_start_h > 0:
                covering.append(
                    -SignedInterval(
                        start=basetime,
                        end=basetime + datetime.timedelta(hours=step_start_h),
                        base=basetime,
                    )
                )
            return check_covering(covering, start, end)

        if self._kind == "from-zero-reset":
            reset = self._hours
            first_cycle = step_start_h // reset * reset
            last_cycle = (step_end_h - 1) // reset * reset
            covering = []
            if step_start_h > first_cycle:
                covering.append(
                    -SignedInterval(
                        start=basetime + datetime.timedelta(hours=first_cycle),
                        end=basetime + datetime.timedelta(hours=step_start_h),
                        base=basetime,
                    )
                )
            for cycle in range(first_cycle, last_cycle + 1, reset):
                covering.append(
                    SignedInterval(
                        start=basetime + datetime.timedelta(hours=cycle),
                        end=basetime + datetime.timedelta(hours=min(cycle + reset, step_end_h)),
                        base=basetime,
                    )
                )
            return check_covering(covering, start, end)

        # increment (a fixed per-step window of length ``L``): tile the
        # requested window with ``L``-long increments and sum them, so a
        # ``period`` coarser than the source increment re-accumulates (e.g. a
        # 6 h window from 3 h increments = two fields). The window must be a
        # whole multiple of ``L``.
        length = self._hours
        window = step_end_h - step_start_h
        if window % length != 0:
            raise ValueError(
                f"accumulate: the requested window ({window}h) must be a whole multiple of the "
                f"source increment ({length}h) to re-accumulate; got a {window}h window."
            )
        covering = [
            SignedInterval(
                start=basetime + datetime.timedelta(hours=k),
                end=basetime + datetime.timedelta(hours=k + length),
                base=basetime,
            )
            for k in range(step_start_h, step_end_h, length)
        ]
        return check_covering(covering, start, end)


class ValidTimeCovering(Covering):
    """Covering for base-less, validity-time-indexed increment source data.

    The source stores one accumulation per fixed-length window ending at its
    own validity time (a flat valid-time index).  The requested window
    ``[start, end]`` is tiled into ``length``-long base-less increments and
    summed; ``end − start`` must be a whole multiple of ``length``.

    There is no search and no midnight alignment: the fields are addressed by
    validity time, so the tiling is exact and anchored on the requested window
    itself (a ``length`` that does not divide 24 h is therefore fine — e.g. a
    5 h source serving a 5 h period).  A source that lacks one of the tiled
    fields fails loudly later, at the completeness check — exactly as a search
    miss would.

    Parameters
    ----------
    length
        The window length each source field holds (a fixed duration).
    """

    def __init__(self, length: datetime.timedelta) -> None:
        self.length = length

    def cover(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        *,
        basetime: datetime.datetime | None = None,
    ) -> list[SignedInterval]:
        if basetime is not None:
            raise NotImplementedError(
                "ValidTimeCovering describes base-less source data and does not honour "
                "an externally-imposed basetime."
            )
        length_seconds = self.length.total_seconds()
        window_seconds = (end - start).total_seconds()
        if window_seconds % length_seconds != 0:
            raise ValueError(
                f"accumulate: the requested window ({window_seconds / 3600:g}h) must be a whole "
                f"multiple of the source increment ({length_seconds / 3600:g}h) to re-accumulate."
            )
        covering: list[SignedInterval] = []
        t = start
        while t < end:
            covering.append(SignedInterval(start=t, end=t + self.length, base=None))
            t += self.length
        return check_covering(covering, start, end)


def covering_factory(
    config,
    source_name: str | None = None,
    source: dict | None = None,
) -> Covering:
    """Build a :class:`Covering` from a recipe ``covering:`` value.

    Two input shapes are accepted:

    - **Discriminator form** (recommended)::

          covering:
              auto: <availability config>     # search-based covering

      ``cycle`` is reserved for future use.

      The ``forecast`` discriminator is intentionally not accepted: the
      trajectory branch is selected implicitly by passing
      ``ForecastDates`` to ``AccumulateSource`` (see
      :class:`ForecastCovering`).

    - **Legacy form** (any non-discriminator value, e.g. a list, ``"auto"``,
      a ``{"mars": ...}`` dict): treated as the value of the ``auto``
      discriminator. Used internally by the back-compat path for the
      deprecated ``availability:`` recipe key.

    Parameters
    ----------
    config
        The recipe value.
    source_name
        Source backend name (for ``"auto"`` discovery).
    source
        Source-specific config (for ``"auto"`` discovery).

    Returns
    -------
    Covering
        The covering strategy built from ``config``.
    """
    if isinstance(config, dict) and len(config) == 1 and next(iter(config)) in ("auto", "cycle", "forecast"):
        kind, value = next(iter(config.items()))
        if kind == "auto":
            availability = interval_generator_factory(value, source_name, source)
            return AutoCovering(availability)
        if kind == "cycle":
            raise NotImplementedError("covering: cycle is not implemented yet.")
        if kind == "forecast":
            raise ValueError(
                "The trajectory branch is selected implicitly by passing "
                "ForecastDates to AccumulateSource — do not declare "
                "'covering: { forecast: ... }' in the recipe. Set "
                "'from: {accumulation: ...}' on the "
                "accumulate block instead."
            )
        raise AssertionError(kind)  # unreachable, keeps mypy happy

    # Legacy form: treat as the value of `auto`.
    availability = interval_generator_factory(config, source_name, source)
    return AutoCovering(availability)
