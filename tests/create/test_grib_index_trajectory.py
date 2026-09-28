# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for ``grib-index`` serving a trajectory (forecast) recipe.

Query construction only — no database is opened. ``_run_requests`` is stubbed
out to capture the ``(valid_dates, request)`` pairs, in the same spirit as
``test_mars_hindcast.py`` capturing MARS request dicts.
"""

import datetime

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.grib_index import GribIndexSource


class _Context:
    def trace(self, *args, **kwargs):
        pass


def _source(**kwargs):
    source = GribIndexSource(_Context(), indexdb="/nonexistent.db", **kwargs)
    captured: list = []

    def capture(full_requests):
        captured.extend(full_requests)
        return []

    source._run_requests = capture
    source.captured = captured
    return source


def test_forecast_dates_address_the_run_and_validity_time() -> None:
    """A trajectory row is addressed by the run's date/time plus the validity time."""
    bt = datetime.datetime(2020, 10, 1, 3)
    source = _source(param="2t", levtype="sfc")

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=h), bt) for h in (0, 3, 6)]))

    assert [(d, r["date"], r["time"]) for d, r in source.captured] == [
        ([bt], 20201001, 300),
        ([bt + datetime.timedelta(hours=3)], 20201001, 300),
        ([bt + datetime.timedelta(hours=6)], 20201001, 300),
    ]
    # The recipe's own selectors are preserved alongside the addressing keys.
    assert source.captured[0][1]["param"] == "2t"
    assert source.captured[0][1]["levtype"] == "sfc"


def test_step_is_not_queried() -> None:
    """``step`` is left out of the query: the run plus validity time imply it.

    Its stored encoding is not dependable — eccodes reports a step whose GRIB
    unit is minutes as ``"0m"``, and ``retrieve`` stringifies its criteria, so
    querying ``step=0`` against a stored ``"0m"`` would match nothing silently.
    """
    bt = datetime.datetime(2020, 10, 1, 0)
    source = _source(param="2t")

    source.execute_forecast_dates(ForecastDates([(bt, bt), (bt + datetime.timedelta(hours=3), bt)]))

    assert all("step" not in request for _, request in source.captured)


def test_sub_hourly_step_is_supported() -> None:
    """With no integer step in the query, a sub-hourly lead time is just a validity time."""
    bt = datetime.datetime(2020, 10, 1, 0)
    source = _source()

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(minutes=30), bt)]))

    ((dates, request),) = source.captured
    assert dates == [datetime.datetime(2020, 10, 1, 0, 30)]
    assert (request["date"], request["time"]) == (20201001, 0)


def test_time_uses_the_mars_integer_convention() -> None:
    """``time`` is 0/300/1200, matching what the indexer stored.

    A zero-padded "0300" would be stringified to a value no row carries, and
    ``retrieve`` would silently return nothing rather than raise.
    """
    got = []
    for hh in (0, 3, 6, 12, 18):
        bt = datetime.datetime(2020, 10, 1, hh)
        source = _source()
        source.execute_forecast_dates(ForecastDates([(bt, bt)]))
        got.append(source.captured[0][1]["time"])

    assert got == [0, 300, 600, 1200, 1800]


def test_step_crossing_midnight_keeps_the_runs_own_date() -> None:
    """A step past 00Z keeps the run's date — the row is not re-dated."""
    bt = datetime.datetime(2020, 10, 1, 18)
    source = _source()

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=12), bt)]))

    ((dates, request),) = source.captured
    assert (request["date"], request["time"]) == (20201001, 1800)
    assert dates == [datetime.datetime(2020, 10, 2, 6)]


def test_row_addressing_overrides_a_pinned_run() -> None:
    """The row's run wins over a stale ``date``/``time``/``step`` in the recipe."""
    bt = datetime.datetime(2020, 10, 1, 3)
    source = _source(date=19990101, time=0, step=99)

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=3), bt)]))

    ((_, request),) = source.captured
    assert (request["date"], request["time"]) == (20201001, 300)
    assert "step" not in request


# ---------------------------------------------------------------------------
# Accumulation intervals: run-anchored vs base-less
# ---------------------------------------------------------------------------


def test_run_anchored_intervals_address_the_run() -> None:
    """A from-zero forecast archive is addressed by run, not by window length.

    ``from-zero`` covers ``[vt-period, vt]`` as ``+a(base->vt) -
    a(base->vt-period)``. Each archived field accumulates from the start of
    the run, so its length is its own end step and says nothing about the
    output window — the run plus the validity time is what identifies it.
    """
    bt = datetime.datetime(2020, 10, 1, 0)
    vt = datetime.datetime(2020, 10, 1, 6)
    period = datetime.timedelta(hours=3)
    source = _source(param="tp", levtype="sfc")

    source.execute_forecast_intervals(
        ForecastIntervals(
            items=[(vt, bt, period)],
            intervals=[
                SignedInterval(bt, vt, base=bt),  # + a(0 -> 6)
                SignedInterval(vt - period, bt, base=bt),  # - a(0 -> 3)
            ],
        )
    )

    assert [(d, r["date"], r["time"]) for d, r in source.captured] == [
        ([vt], 20201001, 0),
        ([datetime.datetime(2020, 10, 1, 3)], 20201001, 0),
    ]
    # No window length is imposed on a run-anchored archive.
    assert all("step" not in r for _, r in source.captured)


def test_base_less_intervals_still_use_the_window_length() -> None:
    """A validity-time-indexed archive keeps being addressed by accumulation length."""
    start = datetime.datetime(2020, 10, 1, 5)
    end = datetime.datetime(2020, 10, 1, 6)
    source = _source(param="tp")

    source.execute_intervals(Intervals([end], [SignedInterval(start, end)]))

    ((dates, request),) = source.captured
    assert dates == [end]
    assert request["step"] == 1
    assert "date" not in request and "time" not in request


def test_reversed_base_less_interval_has_a_positive_length() -> None:
    """A subtracted interval must not query a negative step.

    ``end - start`` is negative for a reversed interval, which would query
    ``step=-1`` and match nothing.
    """
    early = datetime.datetime(2020, 10, 1, 5)
    late = datetime.datetime(2020, 10, 1, 6)
    source = _source(param="tp")

    source.execute_intervals(Intervals([late], [SignedInterval(late, early)]))

    ((_, request),) = source.captured
    assert request["step"] == 1


def test_forecast_intervals_are_not_unpacked_as_pairs() -> None:
    """``ForecastIntervals.items`` are triples; the interval path must not unpack them.

    Without an explicit ``execute_forecast_intervals`` the base class would
    fall through to ``execute_forecast_dates`` and raise "too many values to
    unpack".
    """
    bt = datetime.datetime(2020, 10, 1, 0)
    vt = datetime.datetime(2020, 10, 1, 3)
    source = _source()

    source.execute(
        ForecastIntervals(
            items=[(vt, bt, datetime.timedelta(hours=3))],
            intervals=[SignedInterval(bt, vt, base=bt)],
        )
    )

    assert [(d, r["date"], r["time"]) for d, r in source.captured] == [([vt], 20201001, 0)]


def test_a_mixed_covering_addresses_each_interval_in_its_own_terms() -> None:
    """Run-anchored and base-less intervals can coexist in one covering."""
    bt = datetime.datetime(2020, 10, 1, 0)
    vt = datetime.datetime(2020, 10, 1, 6)
    source = _source(param="tp")

    source.execute_intervals(
        Intervals(
            [vt],
            [
                SignedInterval(bt, vt, base=bt),
                SignedInterval(datetime.datetime(2020, 10, 1, 5), vt),
            ],
        )
    )

    run_anchored, base_less = source.captured
    assert (run_anchored[1]["date"], run_anchored[1]["time"]) == (20201001, 0)
    assert "step" not in run_anchored[1]
    assert base_less[1]["step"] == 1
    assert "date" not in base_less[1]
