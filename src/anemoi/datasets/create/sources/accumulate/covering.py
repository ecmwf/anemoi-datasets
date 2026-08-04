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
``interval_generators.py``) which only describes *what the archive
contains*.

Today there are two concrete strategies:

- :class:`AutoCovering` — search-based; wraps the existing Dijkstra
  over an ``IntervalGenerator``. Used by the archive (validity-date)
  path.
- :class:`ForecastCovering` — basetime-imposed; emits the trivial 1-
  or 2-interval decomposition for trajectory accumulations.

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


class Covering(ABC):
    """Strategy producing a covering of an accumulation window.

    Subclasses implement :meth:`cover` to return the list of
    ``SignedInterval`` objects whose signed sum equals
    ``[start, end]``.
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

    Parameters
    ----------
    availability
        Description of what the archive contains.
    positive_only
        Discard reversed candidates during the search, so the covering is a
        plain tiling instead of a signed decomposition.  Set by the caller for
        non-invertible reductions (``max``/``min``).  Filtering during the
        search — rather than rejecting the result afterwards — matters: the
        search minimises total length, so it can return a ``+9h -3h`` path even
        where a clean positive tiling exists, and the failure would then only
        surface after the retrieval had been paid for.
    """

    def __init__(self, availability: IntervalGenerator, positive_only: bool = False) -> None:
        self.availability = availability
        self.positive_only = positive_only

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
        try:
            return self.availability.covering_intervals(start, end, positive_only=self.positive_only)
        except ValueError as e:
            if not self.positive_only:
                raise
            # The bare search error ("Cannot find coverage of ...") gives no hint that
            # the reduction is what ruled out the subtractive solution.
            raise ValueError(
                f"{e}\n"
                "No forward-only covering exists for this window. A max/min reduction cannot "
                "subtract one archived field from another, so the archive must provide intervals "
                "that tile the window exactly, i.e. per-step values ('from-previous-step'). "
                "Check that 'covering:' describes how *this* parameter is archived: the 'auto' "
                "presets describe precipitation-style layouts, and a parameter such as wind gust "
                "may be stored with different step ranges in the very same class/stream."
            ) from e


_VALID_ACCUMULATIONS = ("from-zero", "from-previous-step")


class ForecastCovering(Covering):
    """Covering for trajectory accumulations.

    The basetime is dictated by the caller (e.g. via ``ForecastDates``);
    no archive search is performed. The covering is the trivial 1- or
    2-interval decomposition determined by the ``accumulation`` flag:

    - ``"from-zero"``: archive stores ``a(0, step)`` accumulations from
      the basetime. The window ``[basetime + sA, basetime + sE]`` is
      built as ``+a(0, sE) − a(0, sA)``.
    - ``"from-previous-step"``: archive stores per-step increments
      ``a(step - period, step)``. The window is the single interval
      ``a(sA, sE)``.

    Parameters
    ----------
    period
        Accumulation window length.
    accumulation
        Either ``"from-zero"`` or ``"from-previous-step"``. There is no
        default — the caller must declare it explicitly.
    positive_only
        Reject ``from-zero``, whose decomposition subtracts one archived field
        from another.  Set by the caller for non-invertible reductions
        (``max``/``min``).
    """

    def __init__(self, period: datetime.timedelta, accumulation: str, positive_only: bool = False) -> None:
        if accumulation not in _VALID_ACCUMULATIONS:
            raise ValueError(f"Invalid accumulation {accumulation!r}; " f"expected one of {_VALID_ACCUMULATIONS}")
        if positive_only and accumulation == "from-zero":
            raise ValueError(
                "'accumulation: from-zero' builds a window by subtracting two archived "
                "fields, which is only defined for an additive reduction. A max/min "
                "reduction needs an archive storing per-step values: use "
                "'accumulation: from-previous-step'."
            )
        self.period = period
        self.accumulation = accumulation
        self.positive_only = positive_only

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

        if self.accumulation == "from-zero":
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
            return covering

        # from-previous-step
        return [
            SignedInterval(
                start=basetime + datetime.timedelta(hours=step_start_h),
                end=basetime + datetime.timedelta(hours=step_end_h),
                base=basetime,
            )
        ]


def validate_tiling(
    intervals: Iterable[SignedInterval],
    start: datetime.datetime,
    end: datetime.datetime,
) -> None:
    """Raise unless ``intervals`` is a gapless, all-positive tiling of ``[start, end]``.

    A signed decomposition is fine for an additive reduction but meaningless for
    ``max``/``min``: those cannot subtract, and an interval reaching outside the
    window would fold in a value from outside it.  ``AutoCovering`` already
    guarantees this when built with ``positive_only``, but
    ``CycleIntervalProvider`` builds its intervals directly and only checks that
    *some* interval starts at ``start`` and *some* ends at ``end`` — it can
    return a set with a hole in the middle.  Hence an explicit check.

    Parameters
    ----------
    intervals
        The covering to validate.
    start, end
        The accumulation window the covering must reproduce exactly.

    Raises
    ------
    ValueError
        If the covering is empty, contains a reversed interval, leaves a gap,
        overlaps, or extends outside ``[start, end]``.
    """
    intervals = list(intervals)
    if not intervals:
        raise ValueError(f"Empty covering for window {start} → {end}.")

    reversed_ = [i for i in intervals if i.length <= 0]
    if reversed_:
        raise ValueError(
            f"Covering of {start} → {end} contains reversed interval(s) "
            f"{reversed_}, which a max/min reduction cannot use."
        )

    ordered = sorted(intervals, key=lambda i: i.start)
    if ordered[0].start != start or ordered[-1].end != end:
        raise ValueError(
            f"Covering of {start} → {end} does not line up with the window: "
            f"it spans {ordered[0].start} → {ordered[-1].end}."
        )
    for previous, following in zip(ordered, ordered[1:]):
        if previous.end != following.start:
            what = "gap" if previous.end < following.start else "overlap"
            raise ValueError(
                f"Covering of {start} → {end} has a {what} between {previous} and {following}; "
                f"a max/min reduction needs the window tiled exactly."
            )


def covering_factory(
    config,
    source_name: str | None = None,
    source: dict | None = None,
    positive_only: bool = False,
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
    positive_only
        Forwarded to :class:`AutoCovering`; set for non-invertible reductions.

    Returns
    -------
    Covering
        The covering strategy built from ``config``.
    """
    _DISCRIMINATORS = ("auto", "cycle")

    if isinstance(config, dict) and len(config) == 1 and next(iter(config)) in ("auto", "cycle", "forecast"):
        kind, value = next(iter(config.items()))
        if kind == "auto":
            availability = interval_generator_factory(value, source_name, source)
            return AutoCovering(availability, positive_only=positive_only)
        if kind == "cycle":
            raise NotImplementedError("covering: cycle is not implemented yet.")
        if kind == "forecast":
            raise ValueError(
                "The trajectory branch is selected implicitly by passing "
                "ForecastDates to AccumulateSource — do not declare "
                "'covering: { forecast: ... }' in the recipe. Set "
                "'accumulation: from-zero | from-previous-step' on the "
                "accumulate block instead."
            )
        raise AssertionError(kind)  # unreachable, keeps mypy happy

    # Legacy form: treat as the value of `auto`.
    availability = interval_generator_factory(config, source_name, source)
    return AutoCovering(availability, positive_only=positive_only)
