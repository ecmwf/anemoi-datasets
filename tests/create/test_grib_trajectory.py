# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``grib`` source's forecast and interval path templates.

Path resolution only — no GRIB files are read. ``_read_fields`` is stubbed
out to capture the paths each request resolves to, in the same spirit as
``test_mars_hindcast.py`` capturing MARS request dicts.
"""

import datetime

import pytest

from anemoi.datasets.create.arguments import ForecastDates
from anemoi.datasets.create.arguments import ForecastIntervals
from anemoi.datasets.create.arguments import Intervals
from anemoi.datasets.create.arguments import ValidDates
from anemoi.datasets.create.intervals import SignedInterval
from anemoi.datasets.create.sources.grib import GribSource

# The KNMI CY46 layout: one directory per model run, one file per forecast
# hour — nothing in the name is the validity time.
KNMI_PATH = "/data/{base_date:strftime(%Y/%m/%d/%H)}/fc{base_date:strftime(%Y%m%d%H)}+{step:int(%03d)}h00m"


class _Context:
    """Minimal context: GribSource only needs ``trace``."""

    def trace(self, *args, **kwargs):
        pass


def _source(path, **kwargs):
    """Build a GribSource whose ``_read_fields`` records resolved paths instead of reading."""
    source = GribSource(_Context(), path, **kwargs)
    seen: list[str] = []

    def capture(paths, sel_kwargs, sel_remapping):
        seen.extend(paths)
        return []

    source._read_fields = capture
    source.captured = seen
    return source


# ---------------------------------------------------------------------------
# ForecastDates — base_date / step
# ---------------------------------------------------------------------------


def test_forecast_dates_substitute_base_date_and_step() -> None:
    bt = datetime.datetime(2020, 10, 1, 3)
    source = _source(KNMI_PATH)

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=h), bt) for h in (0, 3, 6)]))

    assert source.captured == [
        "/data/2020/10/01/03/fc2020100103+000h00m",
        "/data/2020/10/01/03/fc2020100103+003h00m",
        "/data/2020/10/01/03/fc2020100103+006h00m",
    ]


def test_forecast_dates_do_not_cross_product_rows() -> None:
    """Each (valid_time, basetime) pair resolves on its own.

    Two runs three hours apart, each at step 3, must give exactly their own
    two files — not the four of a base_date x step cartesian product, which is
    what batching the pairs into one templated substitution would produce.
    """
    bt1 = datetime.datetime(2020, 10, 1, 0)
    bt2 = datetime.datetime(2020, 10, 1, 3)
    source = _source(KNMI_PATH)

    source.execute_forecast_dates(
        ForecastDates(
            [
                (bt1 + datetime.timedelta(hours=3), bt1),
                (bt2 + datetime.timedelta(hours=6), bt2),
            ]
        )
    )

    assert source.captured == [
        "/data/2020/10/01/00/fc2020100100+003h00m",
        "/data/2020/10/01/03/fc2020100103+006h00m",
    ]


def test_forecast_dates_step_is_whole_hours_from_basetime() -> None:
    bt = datetime.datetime(2020, 10, 1, 12)
    source = _source("/data/{step:int(%d)}")

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=36), bt)]))

    assert source.captured == ["/data/36"]


def test_sub_hourly_step_is_refused_not_truncated() -> None:
    """A 30-minute lead time must not silently resolve to the ``+000h`` file."""
    bt = datetime.datetime(2020, 10, 1, 0)
    source = _source("/data/{step:int(%03d)}")

    with pytest.raises(ValueError, match="expected an integer"):
        source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(minutes=30), bt)]))


def test_sub_hourly_step_via_step_minutes_keyword() -> None:
    """``step_minutes`` addresses an archive whose files are named by the minute."""
    bt = datetime.datetime(2020, 10, 1, 0)
    source = _source("/data/fc+{step_minutes:int(%04d)}")

    source.execute_forecast_dates(
        ForecastDates(
            [
                (bt + datetime.timedelta(minutes=30), bt),
                (bt + datetime.timedelta(hours=1, minutes=30), bt),
            ]
        )
    )

    assert source.captured == ["/data/fc+0030", "/data/fc+0090"]


def test_sub_hourly_step_via_step_seconds_keyword() -> None:
    bt = datetime.datetime(2020, 10, 1, 0)
    source = _source("/data/fc+{step_seconds:int(%d)}")

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(minutes=30), bt)]))

    assert source.captured == ["/data/fc+1800"]


def test_step_minutes_alongside_whole_hour_steps() -> None:
    """The new keywords coexist with ``step``; whole hours are unaffected."""
    bt = datetime.datetime(2020, 10, 1, 0)
    source = _source("/data/{step:int(%03d)}h/{step_minutes:int(%05d)}")

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=36), bt)]))

    assert source.captured == ["/data/036h/02160"]


def test_sub_hourly_step_is_fine_when_the_path_ignores_it() -> None:
    """The refusal is only about ``{step}`` — a base_date-only path still works."""
    bt = datetime.datetime(2020, 10, 1, 0)
    source = _source("/data/{base_date:strftime(%Y%m%d%H)}")

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(minutes=30), bt)]))

    assert source.captured == ["/data/2020100100"]


def test_step_zero_padding_formats() -> None:
    """``step`` takes a printf integer format, so the archive's padding can be matched.

    The width is a minimum: a step that outgrows it widens rather than being
    truncated, so an archive whose steps pass 99 keeps working.
    """
    bt = datetime.datetime(2020, 10, 1, 0)
    hours = (0, 1, 12, 123)

    for fmt, expected in (
        ("%d", ["0", "1", "12", "123"]),
        ("%02d", ["00", "01", "12", "123"]),
        ("%03d", ["000", "001", "012", "123"]),
    ):
        source = _source("/data/{step:int(" + fmt + ")}")
        source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=h), bt) for h in hours]))
        assert source.captured == [f"/data/{e}" for e in expected], fmt


def test_forecast_dates_date_keyword_is_the_validity_time() -> None:
    """``date`` keeps meaning the validity time in a forecast request."""
    bt = datetime.datetime(2020, 10, 1, 21)
    source = _source("/data/{date:strftime(%Y%m%d%H)}")

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=6), bt)]))

    assert source.captured == ["/data/2020100203"]


# ---------------------------------------------------------------------------
# Intervals — start_date / end_date / middle_date
# ---------------------------------------------------------------------------


def test_intervals_expose_window_keywords() -> None:
    start = datetime.datetime(2020, 10, 1, 0)
    end = datetime.datetime(2020, 10, 1, 6)
    source = _source(
        "/data/{start_date:strftime(%Y%m%d%H)}-{middle_date:strftime(%Y%m%d%H)}-{end_date:strftime(%Y%m%d%H)}"
    )

    source.execute_intervals(Intervals([end], [SignedInterval(start, end)]))

    assert source.captured == ["/data/2020100100-2020100103-2020100106"]


def test_intervals_date_keyword_is_the_window_end() -> None:
    """``date`` is the archived field's own validity time, i.e. the window end."""
    start = datetime.datetime(2020, 10, 1, 0)
    end = datetime.datetime(2020, 10, 1, 6)
    source = _source("/data/{date:strftime(%Y%m%d%H)}")

    source.execute_intervals(Intervals([end], [SignedInterval(start, end)]))

    assert source.captured == ["/data/2020100106"]


def test_negative_intervals_are_normalised() -> None:
    """A reversed (negative-sign) interval still resolves start < middle < end."""
    early = datetime.datetime(2020, 10, 1, 0)
    late = datetime.datetime(2020, 10, 1, 6)
    source = _source("/data/{start_date:strftime(%H)}-{middle_date:strftime(%H)}-{end_date:strftime(%H)}")

    source.execute_intervals(Intervals([late], [SignedInterval(late, early)]))

    assert source.captured == ["/data/00-03-06"]


def test_intervals_are_read_once() -> None:
    """A repeated interval is resolved once.

    Overlapping accumulation windows routinely need the same archive
    interval; delivering its field twice would trip the accumulator's
    "already done" guard.
    """
    start = datetime.datetime(2020, 10, 1, 0)
    end = datetime.datetime(2020, 10, 1, 6)
    interval = SignedInterval(start, end)
    source = _source("/data/{end_date:strftime(%Y%m%d%H)}")

    source.execute_intervals(Intervals([end, end], [interval, SignedInterval(start, end)]))

    assert source.captured == ["/data/2020100106"]


def test_added_and_subtracted_interval_is_read_once() -> None:
    """A signed pair of the same archive interval resolves to one read.

    ``from-zero`` covers a window as ``+a(base->step) - a(base->step-period)``,
    so one archive field is added to one row and subtracted from another. The
    two ``SignedInterval`` objects differ (their ``start``/``end`` are
    swapped) but denote the *same* field, so fetching both would hand the
    accumulator a copy with no consumer -> "Field not used for any
    accumulation".
    """
    base = datetime.datetime(2020, 10, 1, 0)
    end = datetime.datetime(2020, 10, 1, 3)
    source = _source(KNMI_PATH)

    source.execute_forecast_intervals(
        ForecastIntervals(
            items=[(end, base, datetime.timedelta(hours=3))],
            intervals=[
                SignedInterval(base, end, base=base),  # + a(0 -> 3)
                SignedInterval(end, base, base=base),  # - a(0 -> 3), reversed
            ],
        )
    )

    assert source.captured == ["/data/2020/10/01/00/fc2020100100+003h00m"]


def test_same_window_from_different_runs_are_both_read() -> None:
    """Dedup keys on the base too — same clock window, two runs, two files."""
    start = datetime.datetime(2020, 10, 1, 3)
    end = datetime.datetime(2020, 10, 1, 6)
    source = _source(KNMI_PATH)

    source.execute_forecast_intervals(
        ForecastIntervals(
            items=[(end, start, datetime.timedelta(hours=3))],
            intervals=[
                SignedInterval(start, end, base=datetime.datetime(2020, 10, 1, 0)),
                SignedInterval(start, end, base=datetime.datetime(2020, 10, 1, 3)),
            ],
        )
    )

    assert source.captured == [
        "/data/2020/10/01/00/fc2020100100+006h00m",
        "/data/2020/10/01/03/fc2020100103+003h00m",
    ]


# ---------------------------------------------------------------------------
# ForecastIntervals
# ---------------------------------------------------------------------------


def test_forecast_intervals_expose_base_date_and_step() -> None:
    bt = datetime.datetime(2020, 10, 1, 0)
    end = bt + datetime.timedelta(hours=6)
    period = datetime.timedelta(hours=3)
    source = _source(KNMI_PATH)

    source.execute_forecast_intervals(
        ForecastIntervals(
            items=[(end, bt, period)],
            intervals=[SignedInterval(end - period, end, base=bt)],
        )
    )

    assert source.captured == ["/data/2020/10/01/00/fc2020100100+006h00m"]


def test_base_less_forecast_intervals_have_no_step() -> None:
    """A base-less interval cannot define a step, so ``{step}`` is unresolvable."""
    end = datetime.datetime(2020, 10, 1, 6)
    period = datetime.timedelta(hours=3)
    source = _source("/data/{step:int(%d)}")

    with pytest.raises(ValueError, match="Missing parameter 'step'"):
        source.execute_forecast_intervals(
            ForecastIntervals(
                items=[(end, datetime.datetime(2020, 10, 1, 0), period)],
                intervals=[SignedInterval(end - period, end)],
            )
        )


# ---------------------------------------------------------------------------
# Interaction with the recipe's own selectors
# ---------------------------------------------------------------------------


def test_valid_dates_still_batch_every_date() -> None:
    """Regression: the non-forecast path resolves all dates in one substitution."""
    source = _source("/data/{date:strftime(%Y%m%d)}")

    source.execute_valid_dates(
        ValidDates([datetime.datetime(2020, 10, 1), datetime.datetime(2020, 10, 2)]),
    )

    assert source.captured == ["/data/20201001", "/data/20201002"]


def test_recipe_selector_wins_over_template_keyword() -> None:
    """An explicit ``step:`` selector pins the value substituted into the path.

    The two would otherwise collide as duplicate keyword arguments.
    """
    bt = datetime.datetime(2020, 10, 1, 0)
    source = _source("/data/{step:int(%03d)}", step=12)

    source.execute_forecast_dates(ForecastDates([(bt + datetime.timedelta(hours=6), bt)]))

    assert source.captured == ["/data/012"]


def test_mars_interpolation_keys_are_rejected_on_every_path() -> None:
    bt = datetime.datetime(2020, 10, 1, 0)
    end = bt + datetime.timedelta(hours=6)
    period = datetime.timedelta(hours=3)

    for call in (
        lambda s: s.execute_valid_dates(ValidDates([bt])),
        lambda s: s.execute_forecast_dates(ForecastDates([(end, bt)])),
        lambda s: s.execute_intervals(Intervals([end], [SignedInterval(bt, end)])),
        lambda s: s.execute_forecast_intervals(
            ForecastIntervals(items=[(end, bt, period)], intervals=[SignedInterval(bt, end, base=bt)])
        ),
    ):
        source = _source(KNMI_PATH, grid="20./20.")
        with pytest.raises(ValueError, match="MARS interpolation parameter"):
            call(source)
