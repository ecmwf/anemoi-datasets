# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest
import xarray as xr

from anemoi.datasets.create.sources.xarray_support.coordinates import DateCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import EnsembleCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LatitudeCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LevelCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LongitudeCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import ScalarCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import StepCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import TimeCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import UnsupportedCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import XCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import YCoordinate
from anemoi.datasets.create.sources.xarray_support.flavour import DefaultCoordinateGuesser


def create_ds(var_name, standard_name, long_name, units, coord_length=5):
    attrs = {
        k: v for k, v in [("standard_name", standard_name), ("long_name", long_name), ("units", units)] if v is not None
    }

    ds = xr.Dataset(
        {"x_wind": ([var_name], np.random.rand(coord_length))},
        coords={
            var_name: xr.DataArray(np.arange(coord_length), dims=var_name, attrs=attrs),
        },
    )
    return ds


@pytest.mark.parametrize(
    "var_name, standard_name, long_name, units, result",
    [
        # longitude
        ("longitude", None, None, None, LongitudeCoordinate),
        ("longitude", None, "longitude", "degrees_east", LongitudeCoordinate),
        ("longitude", None, "longitude", "degrees", LongitudeCoordinate),
        ("lons", "longitude", None, "degrees", LongitudeCoordinate),
        ("lons", None, None, None, UnsupportedCoordinate),
        # latitude
        ("latitude", None, None, None, LatitudeCoordinate),
        ("latitude", None, "latitude", "degrees_north", LatitudeCoordinate),
        ("latitude", None, "latitude", "degrees", LatitudeCoordinate),
        ("lats", "latitude", None, "degrees", LatitudeCoordinate),
        ("lats", None, None, "'degrees", UnsupportedCoordinate),
        # x
        ("x", None, None, None, XCoordinate),
        ("x_coord", "projection_x_coordinate", None, None, XCoordinate),
        ("x_coord", "grid_longitude", None, None, XCoordinate),
        # y
        ("y", None, None, None, YCoordinate),
        ("y_coord", "projection_y_coordinate", None, None, YCoordinate),
        ("y_coord", "grid_latitude", None, None, YCoordinate),
        # time
        ("time", "time", None, None, TimeCoordinate),
        ("time", None, None, None, TimeCoordinate),
        # date
        ("t", "forecast_reference_time", None, None, DateCoordinate),
        ("forecast_reference_time", None, None, None, DateCoordinate),
        ("forecast_reference_time", "forecast_reference_time", None, None, DateCoordinate),
        # step
        ("fp", "forecast_period", None, None, StepCoordinate),
        ("forecast_period", None, "time elapsed since the start of the forecast", None, StepCoordinate),
        ("prediction_timedelta", None, None, None, StepCoordinate),
        # level
        ("lev", "atmosphere_hybrid_sigma_pressure_coordinate", None, None, LevelCoordinate),
        ("h", None, "height", "m", LevelCoordinate),
        ("level", "air_pressure", None, "hPa", LevelCoordinate),
        ("pressure_0", None, "pressure", "hPa", LevelCoordinate),
        ("pressure_0", None, "pressure", "Pa", LevelCoordinate),
        ("level", None, None, None, LevelCoordinate),
        ("lev", None, "level", None, UnsupportedCoordinate),
        ("vertical", "vertical", None, "hPa", LevelCoordinate),
        ("depth", "depth", None, "m", LevelCoordinate),
        ("depth", "depth", None, None, LevelCoordinate),
        ("model_level_number", "model_level_number", None, None, LevelCoordinate),
        # number
        ("realization", None, None, None, EnsembleCoordinate),
        ("number", None, None, None, EnsembleCoordinate),
    ],
)
def test_coordinate_guesser(var_name, standard_name, long_name, units, result):
    ds = create_ds(var_name, standard_name, long_name, units)
    guesser = DefaultCoordinateGuesser(ds)
    guess = guesser.guess(ds[var_name], var_name)
    assert isinstance(guess, result)


def test_coordinate_guesser_scalar():
    var_name = "height"
    ds = create_ds(var_name, None, None, "m", coord_length=1)
    guesser = DefaultCoordinateGuesser(ds)
    guess = guesser.guess(ds[var_name], var_name)
    assert isinstance(guess, ScalarCoordinate)
