# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``average`` / ``minimum`` / ``maximum`` time-reduction sources."""

import datetime
import os

import numpy as np
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.datasets import open_dataset
from anemoi.datasets.create.recipe import Recipe
from anemoi.datasets.create.sources.reduce_support import ReduceSchema
from anemoi.datasets.create.sources.reduce_support import window_samples

from .utils.create import create_dataset

# ── the window ───────────────────────────────────────────────────────


def test_window_is_end_anchored_and_half_open() -> None:
    """A 24 h window over 6-hourly data is the four samples ending at the date."""
    date = datetime.datetime(2021, 1, 2)
    samples = window_samples(date, datetime.timedelta(hours=24), datetime.timedelta(hours=6))
    assert samples == [
        datetime.datetime(2021, 1, 1, 6),
        datetime.datetime(2021, 1, 1, 12),
        datetime.datetime(2021, 1, 1, 18),
        datetime.datetime(2021, 1, 2, 0),
    ]
    # The start of the window belongs to the previous window, as for accumulate.
    assert date - datetime.timedelta(hours=24) not in samples


def test_window_of_one_sample() -> None:
    date = datetime.datetime(2021, 1, 2)
    assert window_samples(date, datetime.timedelta(hours=6), datetime.timedelta(hours=6)) == [date]


def test_window_rejects_a_frequency_that_does_not_divide_the_period() -> None:
    with pytest.raises(ValueError, match="must divide the requested 'period'"):
        window_samples(datetime.datetime(2021, 1, 2), datetime.timedelta(hours=5), datetime.timedelta(hours=2))


# ── the recipe schema ────────────────────────────────────────────────

SOURCE = {"mars": {"class": "ea", "type": "an", "levtype": "sfc", "param": ["2t"]}}


def test_schema_accepts_the_documented_shape() -> None:
    schema = ReduceSchema.model_validate({"period": "24h", "from": {"frequency": "6h"}, "source": SOURCE})
    assert schema.period == datetime.timedelta(hours=24)
    assert schema.from_.frequency == datetime.timedelta(hours=6)


def test_schema_requires_from() -> None:
    with pytest.raises(ValueError, match="'from:' is required"):
        ReduceSchema.model_validate({"period": "24h", "source": SOURCE})


@pytest.mark.parametrize("from_", [{"accumulation": "6h"}, {"lookup-table": {}}])
def test_schema_rejects_the_accumulate_vocabulary(from_: dict) -> None:
    with pytest.raises(ValueError, match="belong to 'accumulate:'"):
        ReduceSchema.model_validate({"period": "24h", "from": from_, "source": SOURCE})


def test_schema_accepts_the_run_anchored_shape() -> None:
    schema = ReduceSchema.model_validate(
        {"period": "6h", "from": {"base_dates": True, "frequency": "1h"}, "source": SOURCE}
    )
    assert schema.is_run_anchored
    assert schema.from_.frequency == datetime.timedelta(hours=1)


def test_accumulate_from_layout_spelling_is_still_accepted() -> None:
    """Undocumented, silent: ``accumulate``'s sentinel folds into the flag."""
    schema = ReduceSchema.model_validate(
        {"period": "6h", "from": {"base_dates": "from-layout", "frequency": "1h"}, "source": SOURCE}
    )
    assert schema.is_run_anchored
    assert schema.from_.base_dates is True


def test_schema_rejects_base_dates_false() -> None:
    """A one-valued flag invites ``false``; say what it should have been."""
    with pytest.raises(ValueError, match="omit 'base_dates' entirely"):
        ReduceSchema.model_validate(
            {"period": "6h", "from": {"base_dates": False, "frequency": "1h"}, "source": SOURCE}
        )


def test_schema_rejects_an_explicit_base_dates_table() -> None:
    """A *different* forecast archive needs run selection; postponed."""
    with pytest.raises(ValueError, match="not supported\n?\s*yet|only accepts 'true'"):
        ReduceSchema.model_validate(
            {"period": "6h", "from": {"base_dates": {"times": [0, 12]}, "frequency": "1h"}, "source": SOURCE}
        )


def test_schema_rejects_steps_in_from() -> None:
    """The lead times are derived from the output steps, never declared."""
    with pytest.raises(ValueError, match="does not take 'steps'"):
        ReduceSchema.model_validate(
            {
                "period": "6h",
                "from": {
                    "base_dates": True,
                    "steps": {"start": "1h", "end": "12h", "frequency": "1h"},
                    "frequency": "1h",
                },
                "source": SOURCE,
            }
        )


def test_schema_requires_a_frequency() -> None:
    with pytest.raises(ValueError, match="needs a 'frequency:'"):
        ReduceSchema.model_validate({"period": "6h", "from": {"base_dates": True}, "source": SOURCE})


def test_schema_rejects_a_period_that_is_not_whole_samples() -> None:
    with pytest.raises(ValueError, match="must divide the requested 'period'"):
        ReduceSchema.model_validate({"period": "5h", "from": {"frequency": "2h"}, "source": SOURCE})


def test_schema_rejects_the_deprecated_covering_key() -> None:
    with pytest.raises(ValueError):
        ReduceSchema.model_validate(
            {"period": "24h", "from": {"frequency": "6h"}, "source": SOURCE, "covering": {"auto": "6h"}}
        )


@pytest.mark.parametrize("name", ["average", "minimum", "maximum"])
def test_recipe_accepts_every_spelling(name: str) -> None:
    recipe = Recipe(
        **{
            "dates": {"start": "2021-01-01", "end": "2021-01-03", "frequency": "24h"},
            "input": {name: {"period": "24h", "from": {"frequency": "6h"}, "source": SOURCE}},
        }
    )
    assert recipe.input is not None


TRAJECTORY = {
    "base_dates": {"start": "2021-01-01", "end": "2021-01-02", "frequency": "12h"},
    "steps": {"start": "6h", "end": "12h", "frequency": "6h"},
    "output": {"layout": "trajectories"},
}
GRIDDED = {"dates": {"start": "2021-01-01", "end": "2021-01-02", "frequency": "6h"}}
RUN_ANCHORED = {"period": "6h", "from": {"base_dates": True, "frequency": "1h"}, "source": SOURCE}
BASE_LESS = {"period": "6h", "from": {"frequency": "1h"}, "source": SOURCE}


def test_run_anchored_from_needs_a_trajectory_layout() -> None:
    """``from-layout`` inherits the run from a layout only the trajectories layout imposes."""
    with pytest.raises(ValueError, match="only valid in a 'layout: trajectories' recipe"):
        Recipe(**{**GRIDDED, "input": {"average": RUN_ANCHORED}})

    Recipe(**{**TRAJECTORY, "input": {"average": RUN_ANCHORED}})


def test_base_less_from_works_in_both_layouts() -> None:
    Recipe(**{**GRIDDED, "input": {"average": BASE_LESS}})
    Recipe(**{**TRAJECTORY, "input": {"average": BASE_LESS}})


def test_run_anchored_window_must_not_straddle_the_basetime() -> None:
    short = {**TRAJECTORY, "steps": {"start": "3h", "end": "12h", "frequency": "3h"}}
    with pytest.raises(ValueError, match="must be >= 'period'"):
        Recipe(**{**short, "input": {"minimum": RUN_ANCHORED}})


def test_base_less_window_may_reach_before_the_basetime() -> None:
    """Deliberately *not* the accumulate rule: analyses exist before the run starts."""
    short = {**TRAJECTORY, "steps": {"start": "3h", "end": "12h", "frequency": "3h"}}
    Recipe(**{**short, "input": {"average": BASE_LESS}})


def test_reduce_is_not_a_recipe_spelling() -> None:
    """``reduce:``/``operation:`` are implementation details, not recipe keys."""
    from anemoi.datasets.create.sources import source_registry

    assert "reduce" not in source_registry.registered

    with pytest.raises(ValueError, match="Unknown source or filter"):
        Recipe(
            **{
                "dates": {"start": "2021-01-01", "end": "2021-01-03", "frequency": "24h"},
                "input": {"reduce": {"operation": "mean", "period": "24h", "source": SOURCE}},
            }
        )


# ── end to end, against the 6-hourly GRIB test data ──────────────────

# grib-20100101.grib / grib-20100102.grib hold 6-hourly fields for the two days.
GRIB_DATES = {"start": "2010-01-01T12:00:00", "end": "2010-01-02T18:00:00", "frequency": "6h"}


def _grib_path(get_test_data) -> str:
    data1 = get_test_data("anemoi-datasets/create/grib-20100101.grib")
    data2 = get_test_data("anemoi-datasets/create/grib-20100102.grib")
    assert os.path.dirname(data1) == os.path.dirname(data2)
    return os.path.join(os.path.dirname(data1), "grib-{date:strftime(%Y%m%d)}.grib")


def _plain(path: str, dates: dict):
    """The raw 6-hourly fields, as the reference to reduce by hand."""
    created = create_dataset(
        recipe={"dates": dates, "input": {"grib": {"path": path}}},
        output=None,
    )
    return open_dataset(created)


def _reduced(name: str, path: str, period: str):
    created = create_dataset(
        recipe={
            "dates": GRIB_DATES,
            "input": {name: {"period": period, "from": {"frequency": "6h"}, "source": {"grib": {"path": path}}}},
        },
        output=None,
    )
    return open_dataset(created)


@skip_if_offline
@pytest.mark.parametrize(
    "name,reduce",
    [
        ("average", lambda a, b: (a + b) / 2),
        ("minimum", np.minimum),
        ("maximum", np.maximum),
    ],
)
def test_reduction_over_two_samples(name: str, reduce, get_test_data: callable) -> None:
    """A 12 h window over 6-hourly data reduces exactly two samples: ``(t-6h, t]``."""
    path = _grib_path(get_test_data)

    # One extra date at the front, so that the raw dataset holds the t-6h
    # sample of the first reduced window.
    raw = _plain(path, {"start": "2010-01-01T06:00:00", "end": "2010-01-02T18:00:00", "frequency": "6h"})
    out = _reduced(name, path, "12h")

    assert out.shape == (len(raw) - 1, *raw.shape[1:])
    assert out.variables == raw.variables

    for i in range(len(out)):
        assert out.dates[i] == raw.dates[i + 1]
        np.testing.assert_allclose(out[i], reduce(raw[i], raw[i + 1]), rtol=1e-6)


@skip_if_offline
@pytest.mark.parametrize("name,process", [("average", "average"), ("minimum", "minimum"), ("maximum", "maximum")])
def test_the_reduced_variable_records_its_process_and_window(name: str, process: str, get_test_data: callable) -> None:
    """The output is stamped with the reduction and the window, not left looking instantaneous."""
    out = _reduced(name, _grib_path(get_test_data), "12h")

    for variable, metadata in out.variables_metadata.items():
        assert metadata["process"] == process, variable
        assert metadata["period"] == ["0h", "12h"], variable


@skip_if_offline
def test_average_of_a_single_sample_window_is_the_field_itself(get_test_data: callable) -> None:
    """``period == from.frequency`` reduces one sample, so the values are unchanged."""
    path = _grib_path(get_test_data)

    raw = _plain(path, GRIB_DATES)
    out = _reduced("average", path, "6h")

    assert out.shape == raw.shape
    for i in range(len(out)):
        assert out.dates[i] == raw.dates[i]
        np.testing.assert_allclose(out[i], raw[i], rtol=1e-6)


@skip_if_offline
def test_base_less_source_fills_a_trajectory_layout(get_test_data: callable) -> None:
    """Case 1 end to end: analyses reduced onto ``(basetime, step)`` rows."""
    path = _grib_path(get_test_data)

    created = create_dataset(
        recipe={
            "base_dates": {"start": "2010-01-01T12:00:00", "end": "2010-01-02T00:00:00", "frequency": "12h"},
            "steps": {"start": "6h", "end": "12h", "frequency": "6h"},
            "output": {"layout": "trajectories"},
            "input": {
                "average": {
                    "period": "12h",
                    "from": {"frequency": "6h"},
                    "source": {"grib": {"path": path}},
                }
            },
        },
        output=None,
    )
    out = open_dataset(created)

    # (base_dates, variables, ensembles, steps, grid)
    assert out.shape[0] == 2 and out.shape[3] == 2
    assert [str(d) for d in out.base_dates] == ["2010-01-01T12:00:00", "2010-01-02T00:00:00"]

    raw = _plain(path, {"start": "2010-01-01T12:00:00", "end": "2010-01-02T12:00:00", "frequency": "6h"})
    at = {str(d): i for i, d in enumerate(raw.dates)}

    # row (basetime 01T12, step 6h) is valid at 18:00, window (12:00, 18:00]
    np.testing.assert_allclose(
        out[0][:, :, 0, :],
        (raw[at["2010-01-01T12:00:00"]] + raw[at["2010-01-01T18:00:00"]]) / 2,
        rtol=1e-6,
    )
    # row (basetime 02T00, step 12h) is valid at 12:00, window (06:00, 12:00]
    np.testing.assert_allclose(
        out[1][:, :, 1, :],
        (raw[at["2010-01-02T06:00:00"]] + raw[at["2010-01-02T12:00:00"]]) / 2,
        rtol=1e-6,
    )

    # The window of a step-6h row with a 12h period straddles the basetime,
    # which a base-less source is allowed to do.
    for metadata in out.variables_metadata.values():
        assert metadata["process"] == "average"
        assert metadata["period"] == ["-6h", "6h"]


@skip_if_offline
def test_window_reaching_before_the_available_data_fails(get_test_data: callable) -> None:
    """A window reaching before the available data fails loudly, it is never averaged short."""
    path = _grib_path(get_test_data)

    # 2010-01-01T00 needs samples back to 2009-12-31T06, which no file holds:
    # the failure comes from the grib source, but the point is that nothing
    # silently produces a mean of the samples that do exist.
    with pytest.raises(Exception):
        create_dataset(
            recipe={
                "dates": {"start": "2010-01-01T00:00:00", "end": "2010-01-01T18:00:00", "frequency": "6h"},
                "input": {
                    "average": {
                        "period": "24h",
                        "from": {"frequency": "6h"},
                        "source": {"grib": {"path": path}},
                    }
                },
            },
            output=None,
        )


# ── completeness, against a source that returns exactly what we tell it ──


class _FakeTime:
    def __init__(self, valid_datetime: datetime.datetime) -> None:
        self._valid_datetime = valid_datetime

    def valid_datetime(self) -> datetime.datetime:
        return self._valid_datetime


class _FakeField:
    """The little a reduction source asks of a field, and nothing more."""

    def __init__(self, valid_datetime: datetime.datetime, param: str, value: float) -> None:
        self.time = _FakeTime(valid_datetime)
        self.param = param
        self.values = np.full(4, value)

    def get(self, collections: str | None = None) -> dict:
        assert collections == "metadata.mars"
        return {
            "param": self.param,
            "date": int(self.time.valid_datetime().strftime("%Y%m%d")),
            "time": int(self.time.valid_datetime().strftime("%H%M")),
            "step": 0,
        }

    def __repr__(self) -> str:
        return f"_FakeField({self.param}, {self.time.valid_datetime()})"


class _FakeContext:
    """A context whose subsource returns a fixed list of fields, and records the ask."""

    def __init__(self, fields: list) -> None:
        self.fields = fields
        self.argument = None

    def create_source(self, source: dict, kind: str, key: str):
        def call(context, argument):
            self.argument = argument
            return self.fields

        return call


def _source(name: str, fields: list, period: str = "12h", from_: dict | None = None):
    from anemoi.datasets.create.sources import source_registry

    return source_registry.lookup(name)(
        _FakeContext(fields),
        source={"grib": {"path": "unused"}},
        period=period,
        **{"from": from_ or {"frequency": "6h"}},
    )


def _dates(*hours: int):
    from anemoi.datasets.create.arguments import ValidDates

    return ValidDates([datetime.datetime(2021, 1, 1, h) for h in hours])


def test_a_window_missing_one_sample_is_an_error() -> None:
    """Reducing what turned up would bias the field and its statistics."""
    fields = [_FakeField(datetime.datetime(2021, 1, 1, 12), "2t", 1.0)]  # 06:00 is missing
    source = _source("average", fields)

    with pytest.raises(ValueError, match="missing source samples"):
        source.execute_valid_dates(_dates(12))


def test_a_variable_missing_for_a_whole_date_is_an_error() -> None:
    """A variable absent for a whole date builds no window at all.

    Completeness alone cannot see that — there is nothing to be incomplete —
    so the (date, variable) grid is checked separately.
    """
    from anemoi.datasets.create.sources.reduce_support.reducer import AverageReducer

    source = _source("average", [])
    dates = _dates(12, 18)

    def complete_reducer(date: datetime.datetime, param: str) -> AverageReducer:
        samples = window_samples(date, source.period, source.frequency)
        reducer = AverageReducer(date, period=source.period, key=(("param", param),), samples=samples)
        for sample in samples:
            reducer.compute(np.zeros(4), sample)
        assert reducer.is_complete()
        return reducer

    reducers = {
        (dates[0], None, (("param", "2t"),)): complete_reducer(dates[0], "2t"),
        (dates[1], None, (("param", "2t"),)): complete_reducer(dates[1], "2t"),
        # msl only ever turned up for the second date.
        (dates[1], None, (("param", "msl"),)): complete_reducer(dates[1], "msl"),
    }

    with pytest.raises(ValueError, match="no source data at all"):
        source._finalise(reducers, [], [(d, None) for d in dates])


def test_a_field_outside_every_window_is_an_error() -> None:
    """A source cadence that does not match ``from.frequency`` is caught, not ignored."""
    fields = [
        _FakeField(datetime.datetime(2021, 1, 1, 6), "2t", 1.0),
        _FakeField(datetime.datetime(2021, 1, 1, 9), "2t", 2.0),  # 3-hourly data, not 6-hourly
        _FakeField(datetime.datetime(2021, 1, 1, 12), "2t", 3.0),
    ]
    source = _source("average", fields)

    with pytest.raises(ValueError, match="not part of any window"):
        source.execute_valid_dates(_dates(12))


def test_the_same_sample_twice_is_an_error() -> None:
    fields = [
        _FakeField(datetime.datetime(2021, 1, 1, 6), "2t", 1.0),
        _FakeField(datetime.datetime(2021, 1, 1, 6), "2t", 1.0),
    ]
    source = _source("average", fields)

    with pytest.raises(ValueError, match="already reduced"):
        source.execute_valid_dates(_dates(12))


def test_a_run_anchored_from_is_refused_in_a_gridded_build() -> None:
    source = _source("average", [], period="6h", from_={"base_dates": True, "frequency": "1h"})
    with pytest.raises(ValueError, match="only 'layout: trajectories' imposes"):
        source.execute_valid_dates(_dates(12))


# ── trajectories: which samples are asked of the subsource ───────────


def _forecast(*items):
    from anemoi.datasets.create.arguments import ForecastDates

    return ForecastDates(list(items))


def test_base_less_trajectory_asks_for_plain_validity_times() -> None:
    """Case 1: the row's basetime only stamps the output; the fetch is base-less."""
    from anemoi.datasets.create.arguments import ValidDates

    basetime = datetime.datetime(2021, 1, 1, 0)
    source = _source("average", [], period="12h", from_={"frequency": "6h"})

    with pytest.raises(ValueError):  # no fields returned; we only inspect the ask
        source.execute_forecast_dates(_forecast((datetime.datetime(2021, 1, 1, 12), basetime)))

    argument = source.context.argument
    assert isinstance(argument, ValidDates)
    assert argument.dates == [datetime.datetime(2021, 1, 1, 6), datetime.datetime(2021, 1, 1, 12)]


def test_run_anchored_lead_times_are_derived_from_the_output_steps() -> None:
    """Case 2: the samples are denser than the output steps and reach below them."""
    from anemoi.datasets.create.arguments import ForecastDates

    basetime = datetime.datetime(2021, 1, 1, 0)
    # Output steps 6h and 12h, a 6h window, hourly source fields.
    rows = [(basetime + datetime.timedelta(hours=h), basetime) for h in (6, 12)]
    source = _source("maximum", [], period="6h", from_={"base_dates": True, "frequency": "1h"})

    with pytest.raises(ValueError):
        source.execute_forecast_dates(_forecast(*rows))

    argument = source.context.argument
    assert isinstance(argument, ForecastDates)
    leads = sorted((vt - bt).total_seconds() / 3600 for vt, bt in argument.items)
    # (0h, 6h] -> 1..6 and (6h, 12h] -> 7..12, all from the same run.
    assert leads == [float(h) for h in range(1, 13)]
    assert {bt for _, bt in argument.items} == {basetime}


def test_run_anchored_window_straddling_the_basetime_is_refused_at_build_time() -> None:
    basetime = datetime.datetime(2021, 1, 1, 0)
    source = _source("average", [], period="12h", from_={"base_dates": True, "frequency": "1h"})

    with pytest.raises(ValueError, match="straddles the basetime"):
        source.execute_forecast_dates(_forecast((basetime + datetime.timedelta(hours=6), basetime)))


def test_the_same_validity_time_from_two_runs_is_not_folded_together() -> None:
    """Two runs reaching the same validity time are two fields, not one sample seen twice."""
    from anemoi.datasets.create.arguments import ForecastDates

    early = datetime.datetime(2021, 1, 1, 0)
    late = datetime.datetime(2021, 1, 1, 6)
    valid = datetime.datetime(2021, 1, 1, 12)
    source = _source("average", [], period="6h", from_={"base_dates": "from-layout", "frequency": "6h"})

    with pytest.raises(ValueError):
        source.execute_forecast_dates(_forecast((valid, early), (valid, late)))

    argument = source.context.argument
    assert isinstance(argument, ForecastDates)
    assert sorted(argument.items) == [(valid, early), (valid, late)]


@pytest.mark.parametrize("name,time_method", [("average", "avg"), ("minimum", "min"), ("maximum", "max")])
def test_each_source_stamps_its_own_time_method(name: str, time_method: str) -> None:
    """``proc.time_method`` is what anemoi-transform reads back as the statistical process."""
    from anemoi.datasets.create.sources import source_registry

    assert source_registry.lookup(name).reducer_class.time_method == time_method
