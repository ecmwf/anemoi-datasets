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

from anemoi.datasets.create.sources.xarray_support.coordinates import DateCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LatitudeCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LevelCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import LongitudeCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import StepCoordinate
from anemoi.datasets.create.sources.xarray_support.coordinates import TimeCoordinate
from anemoi.datasets.create.sources.xarray_support.fieldlist import XarrayFieldList
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

    return XarrayFieldList.from_xarray(ds)


def test_levtype_metadata_on_pressure_level_field(mixed_fieldlist):
    """Fields from a variable with a level coordinate must expose levtype and units in metadata."""
    q_fields = [f for f in mixed_fieldlist if f.metadata("variable") == "q"]
    assert len(q_fields) > 0, "Expected at least one field for variable 'q'"

    for field in q_fields:
        levtype = field.metadata("levtype", default=None)
        assert (
            levtype is not None
        ), f"levtype is None for pressure-level field q at level={field.metadata('level', default='?')}"

        units = field.metadata("units", default=None)
        assert units == "kg kg**-1", f"Expected units='kg kg**-1' for q, got {units!r}"


def test_levtype_metadata_on_surface_field(mixed_fieldlist):
    """Fields from a variable without a level coordinate must default levtype to 'sfc' and carry units."""
    tcw_fields = [f for f in mixed_fieldlist if f.metadata("variable") == "tcw"]
    assert len(tcw_fields) > 0, "Expected at least one field for variable 'tcw'"

    for field in tcw_fields:
        levtype = field.metadata("levtype")
        assert levtype == "sfc", f"Expected levtype='sfc' for surface field tcw, got levtype={levtype!r}"

        units = field.metadata("units")
        assert units == "kg m-2", f"Expected units='kg m-2' for tcw, got {units!r}"


def test_sel_by_levtype_pressure_level(mixed_fieldlist):
    """Selecting by levtype='pl' must return only pressure-level fields."""
    result = mixed_fieldlist.sel(levtype="pl")
    assert len(result) > 0, "sel(levtype='pl') returned no fields"

    for field in result:
        assert field.metadata("levtype") == "pl"
        assert field.metadata("variable") in ("q", "t")


def test_sel_by_levtype_surface(mixed_fieldlist):
    """Selecting by levtype='sfc' must return only surface fields."""
    result = mixed_fieldlist.sel(levtype="sfc")
    assert len(result) > 0, "sel(levtype='sfc') returned no fields"

    for field in result:
        assert field.metadata("levtype") == "sfc"
        assert field.metadata("variable") == "tcw"


def test_sel_by_param_and_levtype(mixed_fieldlist):
    """Selecting by both param and surface level field sel(param=['tcw'], levtype='sfc')."""
    result = mixed_fieldlist.sel(param=["tcw"], levtype="sfc")
    assert len(result) > 0, "sel(param=['tcw'], levtype='sfc') returned no fields"

    for field in result:
        assert field.metadata("variable") == "tcw"
        assert field.metadata("levtype") == "sfc"
        assert field.metadata("units") == "kg m-2"
