# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import pytest
import xarray as xr

from anemoi.datasets.create.sources.xarray_support.coordinates import DateCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LatitudeCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LevelCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LongitudeCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import StepCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import TimeCoordinate
from anemoi.datasets.create.sources.xarray_support.time import ForecastFromValidTimeAndStep
from anemoi.datasets.create.sources.xarray_support.variable import Variable


@pytest.fixture
def sample_variable():
    ds = xr.Dataset(
        {
            "temperature": (("time", "pressure", "latitude", "longitude"), np.ones((1, 3, 3, 4))),
        },
        coords={
            "time": ("time", [datetime.datetime(2025, 2, 3)]),
            "forecast_period": ([0]),
            "forecast_reference_time": ([datetime.datetime(2025, 2, 3)]),
            "pressure": ("pressure", [850, 925, 1000]),
            "latitude": ("latitude", np.linspace(-90, 90, 3)),
            "longitude": ("longitude", np.linspace(-180, 180, 4)),
        },
    )

    time_coordinates = [
        TimeCoordinate(ds["time"]),
        StepCoordinate(ds["forecast_period"]),
        DateCoordinate(ds["forecast_reference_time"]),
    ]

    time = ForecastFromValidTimeAndStep(*time_coordinates)
    coordinates = [
        *time_coordinates,
        LevelCoordinate(ds["pressure"], "pl"),
        LatitudeCoordinate(ds["latitude"]),
        LongitudeCoordinate(ds["longitude"]),
    ]

    variable = Variable(ds=ds, variable=ds["temperature"], coordinates=coordinates, grid=None, time=time, metadata={})
    return variable


def test_sel_single_value(sample_variable):
    result = sample_variable.sel({}, pressure=850)
    print(result.variable["pressure"].values)
    assert result.variable["pressure"].values == 850


def test_sel_multiple_values(sample_variable):
    result = sample_variable.sel({}, pressure=[850, 925])
    assert np.array_equal(result.variable["pressure"].values, np.array([850, 925]))


def test_sel_from_scalar_time(sample_variable):
    result = sample_variable.sel({}, valid_datetime=datetime.datetime(2025, 2, 3))
    assert np.array_equal(
        result.variable["time"].values, np.array([datetime.datetime(2025, 2, 3)], dtype="datetime64[ns]")
    )


def test_sel_multiple_coordinates(sample_variable):
    result = sample_variable.sel({}, valid_datetime=datetime.datetime(2025, 2, 3), pressure=[850, 925])
    assert np.array_equal(result.variable["pressure"].values, np.array([850, 925]))
    assert np.array_equal(
        result.variable["time"].values, np.array([datetime.datetime(2025, 2, 3)], dtype="datetime64[ns]")
    )
