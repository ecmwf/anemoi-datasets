import numpy as np  
import earthkit.data as ekd  
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
LATS = np.array([80., 60., 40., 20., 0., -20., -40., -60., -80.])
LONS = np.arange(0., 360., 20.)


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
    lat2, lon2 = _grid(LATS[::-1], LONS)      # N/S flipped
    ds1 = _make_field(lat1, lon1)
    ds2 = _make_field(lat2, lon2)

    with pytest.raises(ValueError, match="[Ii]ncompatible grid"):
        _check_same_grid(ds1, ds2, "/input/join")


def test_join_longitude_offset_fails():
    lat1, lon1 = _grid(LATS, LONS)
    # Same points, but -180 to 180 convention -> different element order
    lons_shifted = np.where(LONS >= 180., LONS - 360., LONS)
    order = np.argsort(lons_shifted)
    lat2, lon2 = _grid(LATS, lons_shifted[order])
    ds1 = _make_field(lat1, lon1)
    ds2 = _make_field(lat2, lon2)

    with pytest.raises(ValueError, match="[Ii]ncompatible grid"):
        _check_same_grid(ds1, ds2, "/input/join")


def test_join_different_shape_fails():
    lat1, lon1 = _grid(LATS, LONS)
    lat2, lon2 = _grid(LATS, np.arange(0., 360., 10.))  # different resolution
    ds1 = _make_field(lat1, lon1)
    ds2 = _make_field(lat2, lon2)

    with pytest.raises(ValueError, match="[Ii]ncompatible"):
        _check_same_grid(ds1, ds2, "/input/join")