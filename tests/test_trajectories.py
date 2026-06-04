# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the trajectories feature.

Covers:
- ``TrajectoriesOutput.get_chunking()``
- ``Recipe`` model validator (steps required for trajectories layout)
- ``Steps`` iteration / array conversion
- Synthetic ``TrajectoriesZarr`` construction
- ``StepSubset`` and ``SingleStepView`` wrappers
- ``Dataset._step_to_index`` / ``_steps_to_indices`` error paths
- ``FromTrajectoriesSource._inject_steps`` and ``_basetime_matches``
- ``expand_to_by`` step-syntax parsing
"""

import datetime
from unittest.mock import patch

import numpy as np
import pytest
import zarr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_trajectories_zarr(
    n_dates: int = 4,
    n_steps: int = 5,
    n_vars: int = 3,
    n_cells: int = 10,
    step_hours: list[int] | None = None,
    frequency_h: int = 6,
    vars: list[str] | None = None,
) -> zarr.Group:
    """Build a minimal in-memory zarr group with trajectories layout.

    Parameters
    ----------
    n_dates : int
        Number of base dates.
    n_steps : int
        Number of forecast steps.
    n_vars : int
        Number of variables.
    n_cells : int
        Number of grid cells.
    step_hours : list of int, optional
        Explicit step values in hours.  Defaults to ``[6, 12, 18, 24, 30]``.
    frequency_h : int
        Base-date frequency in hours.
    vars : list of str, optional
        Variable names.  Defaults to ``["a", "b", "c"]`` (first *n_vars*).

    Returns
    -------
    zarr.Group
        In-memory zarr group.
    """
    if step_hours is None:
        step_hours = list(range(6, 6 * (n_steps + 1), 6))[:n_steps]
    if vars is None:
        vars = [chr(ord("a") + i) for i in range(n_vars)]

    root = zarr.group()

    start = datetime.datetime(2021, 1, 1)
    dates = np.array(
        [start + datetime.timedelta(hours=frequency_h * i) for i in range(n_dates)],
        dtype="datetime64[s]",
    )
    steps = np.array([np.timedelta64(h, "h") for h in step_hours])

    rng = np.random.default_rng(0)
    data = rng.random((n_dates, n_vars, 1, n_steps, n_cells)).astype("float32")

    root.create_array("data", data=data, chunks=data.shape, compressor=None)
    root.create_array("base_dates", data=dates, compressor=None)
    root.create_array("steps", data=steps, compressor=None)
    root.create_array("latitudes", data=np.linspace(-90, 90, n_cells), compressor=None)
    root.create_array("longitudes", data=np.linspace(0, 360, n_cells), compressor=None)

    root.attrs["layout"] = "trajectories"
    root.attrs["frequency"] = f"{frequency_h}h"
    root.attrs["resolution"] = "test"
    root.attrs["name_to_index"] = {v: i for i, v in enumerate(vars)}
    root.attrs["variables_metadata"] = {v: {} for v in vars}
    root.attrs["data_request"] = {"grid": 1, "area": "g", "param_level": {}}

    return root


def open_trajectories_zarr(group: zarr.Group):
    """Open a zarr group as a ``TrajectoriesZarr`` dataset."""
    from anemoi.datasets.usage.trajectories.store import TrajectoriesZarr

    return TrajectoriesZarr(group, path="test.zarr")


# ---------------------------------------------------------------------------
# TrajectoriesOutput.get_chunking
# ---------------------------------------------------------------------------


class TestTrajectoriesOutputChunking:

    def _make_output(self, chunking=None):
        from anemoi.datasets.create.recipe.output import TrajectoriesOutput

        kwargs = {}
        if chunking is not None:
            kwargs["chunking"] = chunking
        return TrajectoriesOutput(**kwargs)

    def test_default_chunking(self):
        output = self._make_output()
        coords = {
            "base_dates": list(range(10)),
            "steps": list(range(5)),
            "variables": list(range(4)),
            "ensembles": [0],
            "cells": list(range(100)),
        }
        chunks = output.get_chunking(coords)
        # base_dates→1, steps→1, variables→4 (full), ensembles→1, cells→100 (full)
        assert chunks == (1, 1, 4, 1, 100)

    def test_custom_chunking(self):
        output = self._make_output(chunking={"base_dates": 2, "steps": 3, "ensembles": 1})
        coords = {
            "base_dates": list(range(8)),
            "steps": list(range(6)),
            "variables": list(range(4)),
            "ensembles": [0],
            "cells": list(range(50)),
        }
        chunks = output.get_chunking(coords)
        assert chunks == (2, 3, 4, 1, 50)

    def test_unknown_key_raises(self):
        output = self._make_output(chunking={"base_dates": 1, "bad_key": 1, "ensembles": 1})
        coords = {"base_dates": [0], "steps": [0], "variables": [0], "ensembles": [0], "cells": [0]}
        with pytest.raises(ValueError, match="bad_key"):
            output.get_chunking(coords)


# ---------------------------------------------------------------------------
# Recipe model validator — steps required for trajectories layout
# ---------------------------------------------------------------------------


class TestRecipeStepsValidator:

    def _gridded_recipe(self):
        return {
            "dates": {"start": "2021-01-01", "end": "2021-01-02", "frequency": "6h"},
            "input": {"mars": {"param": ["t"]}},
        }

    def _trajectories_recipe(self):
        return {
            "base_dates": {"start": "2021-01-01", "end": "2021-01-02", "frequency": "6h"},
            "input": {"mars": {"param": ["t"]}},
            "output": {"layout": "trajectories"},
            "steps": {"start": "6h", "end": "24h", "frequency": "6h"},
        }

    def test_trajectories_without_steps_raises(self):
        from pydantic import ValidationError

        from anemoi.datasets.create.recipe import Recipe

        recipe_dict = self._trajectories_recipe()
        recipe_dict.pop("steps")

        with pytest.raises(ValidationError, match="steps"):
            Recipe(**recipe_dict)

    def test_trajectories_with_steps_ok(self):
        from anemoi.datasets.create.recipe import Recipe

        r = Recipe(**self._trajectories_recipe())
        assert r.steps is not None
        assert r.base_dates is not None
        assert r.dates is None

    def test_gridded_without_steps_ok(self):
        from anemoi.datasets.create.recipe import Recipe

        r = Recipe(**self._gridded_recipe())
        assert r.steps is None
        assert r.dates is not None
        assert r.base_dates is None

    def test_trajectories_with_dates_raises(self):
        from pydantic import ValidationError

        from anemoi.datasets.create.recipe import Recipe

        recipe_dict = self._trajectories_recipe()
        recipe_dict["dates"] = recipe_dict.pop("base_dates")

        with pytest.raises(ValidationError, match="base_dates"):
            Recipe(**recipe_dict)

    def test_gridded_with_base_dates_raises(self):
        from pydantic import ValidationError

        from anemoi.datasets.create.recipe import Recipe

        recipe_dict = self._gridded_recipe()
        recipe_dict["base_dates"] = recipe_dict.pop("dates")

        with pytest.raises(ValidationError, match="base_dates"):
            Recipe(**recipe_dict)


# ---------------------------------------------------------------------------
# Steps class
# ---------------------------------------------------------------------------


class TestSteps:

    def _make_steps(self, start="6h", end="24h", frequency="6h"):
        from anemoi.datasets.create.trajectories.context import Steps

        return Steps(start=start, end=end, frequency=frequency)

    def test_length(self):
        s = self._make_steps()
        assert len(s) == 4  # 6, 12, 18, 24

    def test_iter(self):
        s = self._make_steps()
        values = list(s)
        assert len(values) == 4

    def test_numpy_array(self):
        s = self._make_steps()
        arr = np.array(s)
        assert arr.shape == (4,)
        assert arr.dtype.kind == "m"  # timedelta

    def test_single_step(self):
        s = self._make_steps(start="6h", end="6h", frequency="6h")
        assert len(s) == 1


# ---------------------------------------------------------------------------
# Synthetic TrajectoriesZarr
# ---------------------------------------------------------------------------


class TestTrajectoriesZarr:

    def setup_method(self):
        self.group = make_trajectories_zarr(n_dates=4, n_steps=5, n_vars=3, n_cells=10)
        self.ds = open_trajectories_zarr(self.group)

    def test_shape(self):
        assert self.ds.shape == (4, 3, 1, 5, 10)

    def test_len(self):
        assert len(self.ds) == 4

    def test_steps_dtype(self):
        assert self.ds.steps.dtype.kind == "m"
        assert len(self.ds.steps) == 5

    def test_start_end_step(self):
        assert self.ds.step_start == datetime.timedelta(hours=6)
        assert self.ds.step_end == datetime.timedelta(hours=30)

    def test_step_frequency(self):
        assert self.ds.step_frequency == datetime.timedelta(hours=6)

    def test_step_frequency_nonuniform(self):
        group = make_trajectories_zarr(step_hours=[6, 12, 24], n_steps=3)
        ds = open_trajectories_zarr(group)
        assert ds.step_frequency is None

    def test_variables(self):
        assert self.ds.variables == ["a", "b", "c"]

    def test_getitem_scalar(self):
        item = self.ds[0]
        assert item.shape == (3, 1, 5, 10)

    def test_getitem_slice(self):
        item = self.ds[0:2]
        assert item.shape == (2, 3, 1, 5, 10)


# ---------------------------------------------------------------------------
# StepSubset
# ---------------------------------------------------------------------------


class TestStepSubset:

    def setup_method(self):
        self.group = make_trajectories_zarr(n_dates=4, n_steps=5, n_vars=3, n_cells=10)
        self.ds = open_trajectories_zarr(self.group)

    def test_shape(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [0, 2])
        assert sub.shape == (4, 3, 1, 2, 10)

    def test_steps_subset(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [0, 2])
        expected = self.ds.steps[[0, 2]]
        np.testing.assert_array_equal(sub.steps, expected)

    def test_start_end_step(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [1, 3])
        assert sub.step_start == datetime.timedelta(hours=12)
        assert sub.step_end == datetime.timedelta(hours=24)

    def test_step_frequency_uniform(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [0, 1, 2])
        assert sub.step_frequency == datetime.timedelta(hours=6)

    def test_step_frequency_nonuniform(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [0, 2, 4])  # 6, 18, 30 h — spacing 12, 12 — uniform
        assert sub.step_frequency == datetime.timedelta(hours=12)

    def test_step_frequency_single(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [0])
        assert sub.step_frequency is None

    def test_getitem_scalar(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [1, 3])
        item = sub[0]
        assert item.shape == (3, 1, 2, 10)

    def test_getitem_slice(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [0, 2])
        item = sub[0:2]
        assert item.shape == (2, 3, 1, 2, 10)

    def test_data_values_match(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        sub = StepSubset(self.ds, [1, 3])
        np.testing.assert_array_equal(sub[0], self.ds[0][:, :, [1, 3], :])


# ---------------------------------------------------------------------------
# SingleStepView
# ---------------------------------------------------------------------------


class TestSingleStepView:

    def setup_method(self):
        self.group = make_trajectories_zarr(n_dates=4, n_steps=5, n_vars=3, n_cells=10)
        self.ds = open_trajectories_zarr(self.group)

    def test_shape(self):
        from anemoi.datasets.usage.trajectories.subset import SingleStepView

        view = SingleStepView(self.ds, 2)
        assert view.shape == (4, 3, 1, 10)

    def test_getitem_scalar(self):
        from anemoi.datasets.usage.trajectories.subset import SingleStepView

        view = SingleStepView(self.ds, 2)
        item = view[0]
        assert item.shape == (3, 1, 10)

    def test_getitem_slice(self):
        from anemoi.datasets.usage.trajectories.subset import SingleStepView

        view = SingleStepView(self.ds, 2)
        item = view[0:2]
        assert item.shape == (2, 3, 1, 10)

    def test_data_values_match(self):
        from anemoi.datasets.usage.trajectories.subset import SingleStepView

        view = SingleStepView(self.ds, 2)
        np.testing.assert_array_equal(view[0], self.ds[0][:, :, 2, :])


# ---------------------------------------------------------------------------
# Dataset._step_to_index / _steps_to_indices (via open_dataset mock)
# ---------------------------------------------------------------------------


class TestStepToIndex:

    def setup_method(self):
        self.group = make_trajectories_zarr(n_dates=4, n_steps=5, n_vars=3, n_cells=10)
        self.ds = open_trajectories_zarr(self.group)

    def test_valid_step(self):
        assert self.ds._step_to_index(6) == 0
        assert self.ds._step_to_index(12) == 1
        assert self.ds._step_to_index(30) == 4

    def test_string_step(self):
        assert self.ds._step_to_index("18") == 2

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError, match="step=999h"):
            self.ds._step_to_index(999)

    def test_steps_to_indices(self):
        assert self.ds._steps_to_indices([6, 18, 30]) == [0, 2, 4]

    def test_steps_to_indices_scalar(self):
        assert self.ds._steps_to_indices(12) == [1]


# ---------------------------------------------------------------------------
# open_dataset(path, step=N) and open_dataset(path, steps=[...])
# ---------------------------------------------------------------------------


class TestOpenDatasetStepSelection:

    def setup_method(self):
        self.group = make_trajectories_zarr(n_dates=4, n_steps=5, n_vars=3, n_cells=10)

    def _open(self, **kwargs):
        from anemoi.datasets import open_dataset

        # Patch zarr.open so open_dataset finds our in-memory group
        with patch("zarr.open", return_value=self.group):
            with patch("anemoi.datasets.usage.store.dataset_lookup", lambda name: name):
                return open_dataset("test", **kwargs)

    def test_single_step_view_shape(self):
        from anemoi.datasets.usage.trajectories.subset import SingleStepView

        ds = open_trajectories_zarr(self.group)
        view = SingleStepView(ds, ds._step_to_index(12))
        assert view.shape == (4, 3, 1, 10)

    def test_step_subset_shape(self):
        from anemoi.datasets.usage.trajectories.subset import StepSubset

        ds = open_trajectories_zarr(self.group)
        sub = StepSubset(ds, ds._steps_to_indices([6, 12]))
        assert sub.shape == (4, 3, 1, 2, 10)


# ---------------------------------------------------------------------------
# expand_to_by — MARS step-syntax parsing
# ---------------------------------------------------------------------------


class TestExpandToBy:

    def _expand(self, x):
        from anemoi.datasets.create.sources.mars.retrieval import expand_to_by

        return expand_to_by(x)

    def test_to_by_syntax(self):
        assert self._expand("6/to/24/by/6") == [6, 12, 18, 24]

    def test_to_syntax(self):
        assert self._expand("6/to/12") == [6, 7, 8, 9, 10, 11, 12]

    def test_slash_list(self):
        result = self._expand("6/18")
        assert result == ["6", "18"]

    def test_single_int(self):
        result = self._expand(6)
        assert result == ["6"]

    def test_single_string(self):
        result = self._expand("12")
        assert result == ["12"]

    def test_list_passthrough(self):
        result = self._expand([6, 12, 18])
        assert result == [6, 12, 18]


# ---------------------------------------------------------------------------
# FromTrajectoriesSource — typed-argument mapping
# ---------------------------------------------------------------------------


class TestFromTrajectoriesSource:
    """Unit tests for FromTrajectoriesSource without touching the inner source."""

    def _make_source(self):
        """Return an instance with __init__ bypassed — safe for testing pure helpers."""
        from unittest.mock import MagicMock

        from anemoi.datasets.create.sources.from_trajectories import FromTrajectoriesSource

        src = object.__new__(FromTrajectoriesSource)
        src.steps_hours = []
        src.bases_pattern = None
        src.inner = MagicMock()
        return src

    # -- _basetime_matches ---------------------------------------------------

    def test_no_pattern_always_matches(self):
        src = self._make_source()
        src.bases_pattern = None
        assert src._basetime_matches(datetime.datetime(2021, 1, 1, 6, 0, 0)) is True

    def test_midnight_pattern_matches_midnight(self):
        src = self._make_source()
        src.bases_pattern = "????-??-?? 00:00:00"
        assert src._basetime_matches(datetime.datetime(2021, 1, 1, 0, 0, 0)) is True

    def test_midnight_pattern_rejects_non_midnight(self):
        src = self._make_source()
        src.bases_pattern = "????-??-?? 00:00:00"
        assert src._basetime_matches(datetime.datetime(2021, 1, 1, 12, 0, 0)) is False

    def test_specific_date_pattern(self):
        src = self._make_source()
        src.bases_pattern = "2021-01-01 *"
        assert src._basetime_matches(datetime.datetime(2021, 1, 1, 6, 0, 0)) is True
        assert src._basetime_matches(datetime.datetime(2021, 1, 2, 0, 0, 0)) is False

    # -- _pick_basetime ------------------------------------------------------

    def test_pick_basetime_no_pattern_picks_first_step(self):
        src = self._make_source()
        src.steps_hours = [6, 12, 18]
        src.bases_pattern = None
        vt = datetime.datetime(2021, 1, 1, 12, 0, 0)
        assert src._pick_basetime(vt) == datetime.datetime(2021, 1, 1, 6, 0, 0)

    def test_pick_basetime_skips_until_match(self):
        src = self._make_source()
        src.steps_hours = [6, 12]
        src.bases_pattern = "????-??-?? 00:00:00"
        # valid 12:00; step 6 → basetime 06:00 (no match); step 12 → 00:00 (match)
        vt = datetime.datetime(2021, 1, 1, 12, 0, 0)
        assert src._pick_basetime(vt) == datetime.datetime(2021, 1, 1, 0, 0, 0)

    def test_pick_basetime_no_match_raises(self):
        src = self._make_source()
        src.steps_hours = [6, 12]
        src.bases_pattern = "????-??-?? 00:00:00"
        # valid 06:00; step 6 → 00:00 matches; change to 09:00 where neither step lands on midnight
        vt = datetime.datetime(2021, 1, 1, 9, 0, 0)
        with pytest.raises(ValueError, match="no basetime matches"):
            src._pick_basetime(vt)

    # -- _as_forecast_dates --------------------------------------------------

    def test_as_forecast_dates_from_list(self):
        from anemoi.datasets.create.arguments import ForecastDates

        src = self._make_source()
        src.steps_hours = [6]
        src.bases_pattern = None
        valid_times = [
            datetime.datetime(2021, 1, 1, 12, 0, 0),
            datetime.datetime(2021, 1, 2, 0, 0, 0),
        ]
        fd = src._as_forecast_dates(valid_times)
        assert isinstance(fd, ForecastDates)
        assert fd.items == [
            (valid_times[0], datetime.datetime(2021, 1, 1, 6, 0, 0)),
            (valid_times[1], datetime.datetime(2021, 1, 1, 18, 0, 0)),
        ]

    def test_execute_delegates_with_forecast_dates(self):
        from anemoi.datasets.create.arguments import ForecastDates

        src = self._make_source()
        src.steps_hours = [6]
        src.bases_pattern = None
        src.inner.execute.return_value = "DATA"

        vt = datetime.datetime(2021, 1, 1, 12, 0, 0)
        out = src.execute([vt])
        assert out == "DATA"
        args, _ = src.inner.execute.call_args
        assert isinstance(args[0], ForecastDates)
        assert args[0].items == [(vt, datetime.datetime(2021, 1, 1, 6, 0, 0))]
