# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import numpy as np
import pytest

from anemoi.datasets.create.input.action import _check_same_grid


def _make_field(latitudes, longitudes):
    """Build a fake earthkit field with the given coordinate ordering."""
    values = np.arange(len(latitudes), dtype=float)
    # earthkit's "numpy" source can carry lat/lon metadata;
    # alternatively use a stub class exposing to_latlon()
    ds = ekd.from_source(
        "numpy",
        values.reshape(1, -1),
        dict(
            latitudes=np.asarray(latitudes),
            longitudes=np.asarray(longitudes),
            param="2t",
            date=20200101,
        ),
    )
    return ds


# Same grid, different ordering: latitude reversed, longitude shifted
LATS = np.array([80.0, 60.0, 40.0, 20.0, 0.0, -20.0, -40.0, -60.0, -80.0])
LONS = np.arange(0.0, 360.0, 20.0)


def _grid(lats, lons):
    """Full meshgrid in the ordering used by the source."""
    la, lo = np.meshgrid(lats, lons)
    return la.ravel(), lo.ravel()


def test_join_same_grid_passes():
    lat1, lon1 = _grid(LATS, LONS)
    ds1 = _make_field(lat1, lon1)
    ds2 = _make_field(lat1, lon1)

    _check_same_grid(ds1, ds2, "/input/join")  # must not raise


def test_join_flipped_latitude_fails():
    lat1, lon1 = _grid(LATS, LONS)
    lat2, lon2 = _grid(LATS[::-1], LONS)  # N/S flipped
    ds1 = _make_field(lat1, lon1)
    ds2 = _make_field(lat2, lon2)

    with pytest.raises(ValueError, match="[Ii]ncompatible grid"):
        _check_same_grid(ds1, ds2, "/input/join")


def test_join_longitude_offset_fails():
    lat1, lon1 = _grid(LATS, LONS)
    # Same points, but -180 to 180 convention -> different element order
    lons_shifted = np.where(LONS >= 180.0, LONS - 360.0, LONS)
    order = np.argsort(lons_shifted)
    lat2, lon2 = _grid(LATS, lons_shifted[order])
    ds1 = _make_field(lat1, lon1)
    ds2 = _make_field(lat2, lon2)

    with pytest.raises(ValueError, match="[Ii]ncompatible grid"):
        _check_same_grid(ds1, ds2, "/input/join")


def test_join_different_shape_fails():
    lat1, lon1 = _grid(LATS, LONS)
    lat2, lon2 = _grid(LATS, np.arange(0.0, 360.0, 10.0))  # different resolution
    ds1 = _make_field(lat1, lon1)
    ds2 = _make_field(lat2, lon2)

    with pytest.raises(ValueError, match="[Ii]ncompatible"):
        _check_same_grid(ds1, ds2, "/input/join")
