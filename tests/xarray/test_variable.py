# (C) Copyright 2025-2026 Anemoi contributors.
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
from earthkit.data.readers.xarray.coordinates import DateCoordinate
from earthkit.data.readers.xarray.coordinates import LatitudeCoordinate
from earthkit.data.readers.xarray.coordinates import LevelCoordinate
from earthkit.data.readers.xarray.coordinates import LongitudeCoordinate
from earthkit.data.readers.xarray.coordinates import StepCoordinate
from earthkit.data.readers.xarray.coordinates import TimeCoordinate
from earthkit.data.readers.xarray.fieldlist import XArrayFieldList
from earthkit.data.readers.xarray.time import ForecastFromValidTimeAndStep
from earthkit.data.readers.xarray.variable import Variable


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


@pytest.fixture
def mixed_fieldlist():
    """Create a mock xarray Dataset with both pressure-level and surface variables."""
    nlev, nlat, nlon, ntime = 3, 4, 5, 2
    times = [datetime.datetime(2025, 6, 4), datetime.datetime(2025, 6, 4, 6)]
    levels = [850, 925, 1000]

    ds = xr.Dataset(
        {
            # Pressure-level variable
            "q": (("level", "latitude", "longitude", "time"), np.ones((nlev, nlat, nlon, ntime))),
            "t": (("level", "latitude", "longitude", "time"), np.ones((nlev, nlat, nlon, ntime))),
            # Surface variable (no level dimension)
            "tcw": (("latitude", "longitude", "time"), np.ones((nlat, nlon, ntime))),
        },
        coords={
            "level": ("level", levels),
            "time": ("time", times),
            "latitude": ("latitude", np.linspace(-90, 90, nlat)),
            "longitude": ("longitude", np.linspace(-180, 180, nlon)),
        },
        attrs={"Conventions": "CF-1.8"},
    )

    # Set variable attributes as a real CF-compliant NetCDF would have
    ds["q"].attrs.update({"units": "kg kg**-1", "standard_name": "specific_humidity", "long_name": "Specific humidity"})
    ds["t"].attrs.update({"units": "K", "standard_name": "air_temperature", "long_name": "Temperature"})
    ds["tcw"].attrs.update(
        {
            "units": "kg m-2",
            "standard_name": "atmosphere_mass_content_of_water_vapor",
            "long_name": "Total precipitable water",
        }
    )

    return XArrayFieldList.from_xarray(ds)


def test_levtype_metadata_on_pressure_level_field(mixed_fieldlist):
    """Fields from a variable with a level coordinate must expose level_type and units in metadata."""
    q_fields = [f for f in mixed_fieldlist if f.get("parameter.variable", default=None) == "q"]
    assert len(q_fields) > 0, "Expected at least one field for variable 'q'"

    for field in q_fields:
        level_type = field.get("vertical.level_type", default=None)
        assert (
            level_type is not None
        ), f"level_type is None for pressure-level field q at level={field.get('vertical.level', default='?')}"

        units = field.get("parameter.units", default=None)
        assert units is not None, "Expected units to be set for q, got None"
        # kg kg**-1 is dimensionless in SI units
        assert str(units) == "dimensionless", f"Expected dimensionless for q (kg/kg), got {str(units)!r}"


def test_levtype_metadata_on_surface_field(mixed_fieldlist):
    """Fields from a variable without a level coordinate must have unknown level_type and carry units."""
    tcw_fields = [f for f in mixed_fieldlist if f.get("parameter.variable", default=None) == "tcw"]
    assert len(tcw_fields) > 0, "Expected at least one field for variable 'tcw'"

    for field in tcw_fields:
        level_type = field.get("vertical.level_type", default=None)
        assert (
            level_type == "unknown"
        ), f"Expected level_type='unknown' for surface field tcw, got level_type={level_type!r}"

        units = field.get("parameter.units", default=None)
        assert units is not None, "Expected units to be set for tcw, got None"
        assert str(units) == "kilogram / meter ** 2", f"Expected 'kilogram / meter ** 2' for tcw, got {str(units)!r}"


def test_sel_by_levtype_pressure_level(mixed_fieldlist):
    """Selecting by vertical.level_type='pressure' must return only pressure-level fields."""
    result = mixed_fieldlist.sel(**{"vertical.level_type": "pressure"})
    assert len(result) > 0, "sel(vertical.level_type='pressure') returned no fields"

    for field in result:
        assert field.get("vertical.level_type") == "pressure"
        assert field.get("parameter.variable") in ("q", "t")


def test_sel_by_levtype_surface(mixed_fieldlist):
    """Selecting by vertical.level_type='unknown' must return only surface fields."""
    result = mixed_fieldlist.sel(**{"vertical.level_type": "unknown"})
    assert len(result) > 0, "sel(vertical.level_type='unknown') returned no fields"

    for field in result:
        assert field.get("vertical.level_type") == "unknown"
        assert field.get("parameter.variable") == "tcw"


def test_sel_by_param_and_levtype(mixed_fieldlist):
    """Selecting by both param and surface level field sel(parameter.variable=['tcw'], vertical.level_type='unknown')."""
    result = mixed_fieldlist.sel(**{"parameter.variable": ["tcw"], "vertical.level_type": "unknown"})
    assert len(result) > 0, "sel(parameter.variable=['tcw'], vertical.level_type='unknown') returned no fields"

    for field in result:
        assert field.get("parameter.variable") == "tcw"
        assert field.get("vertical.level_type") == "unknown"
        units = field.get("parameter.units", default=None)
        assert str(units) == "kilogram / meter ** 2"
