# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import numpy as np

from .coordinates import LatitudeCoordinate
from .coordinates import LevelCoordinate
from .coordinates import LongitudeCoordinate
from .coordinates import ScalarCoordinate
from .coordinates import StepCoordinate
from .coordinates import TimeCoordinate
from .coordinates import XCoordinate
from .coordinates import YCoordinate
from .grid import MeshedGrid
from .grid import UnstructuredGrid


class CoordinateGuesser:

    def __init__(self, ds, flavour=None):
        self.ds = ds
        self._cache = {}

    def guess(self, c, coord):
        if coord not in self._cache:
            self._cache[coord] = self._guess(c, coord)
        return self._cache[coord]

    def _guess(self, c, coord):

        assert c.name == coord

        standard_name = getattr(c, "standard_name", "").lower()
        axis = getattr(c, "axis", "")
        long_name = getattr(c, "long_name", "").lower()
        coord_name = getattr(c, "name", "")
        units = getattr(c, "units", "")

        d = self._is_longitude(
            c,
            axis=axis,
            coord_name=coord_name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_latitude(
            c,
            axis=axis,
            coord_name=coord_name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_x(
            c,
            axis=axis,
            coord_name=coord_name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_y(
            c,
            axis=axis,
            coord_name=coord_name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_time(
            c,
            axis=axis,
            coord_name=coord_name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_step(
            c,
            axis=axis,
            coord_name=coord_name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_level(
            c,
            axis=axis,
            coord_name=coord_name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        if isinstance(c.values, np.ndarray) and c.shape in ((1,), tuple()):
            return ScalarCoordinate(c)

        raise NotImplementedError(
            f"Coordinate {coord} not supported\n{axis=}, {coord_name=},"
            f" {long_name=}, {standard_name=}, units\n\n{c}\n\n{type(c.values)} {c.shape}"
        )

    def _is_longitude(self, c, *, axis, coord_name, long_name, standard_name, units):
        if standard_name == "longitude":
            return LongitudeCoordinate(c)

        if long_name == "longitude" and units == "degrees_east":
            return LongitudeCoordinate(c)

    def _is_latitude(self, c, *, axis, coord_name, long_name, standard_name, units):
        if standard_name == "latitude":
            return LatitudeCoordinate(c)

        if long_name == "latitude" and units == "degrees_north":
            return LatitudeCoordinate(c)

    def _is_x(self, c, *, axis, coord_name, long_name, standard_name, units):
        if standard_name == "projection_x_coordinate":
            return XCoordinate(c)

    def _is_y(self, c, *, axis, coord_name, long_name, standard_name, units):
        if standard_name == "projection_y_coordinate":
            return YCoordinate(c)

    def _is_time(self, c, *, axis, coord_name, long_name, standard_name, units):
        if standard_name == "time":
            return TimeCoordinate(c)

        if coord_name == "time":
            return TimeCoordinate(c)

    def _is_step(self, c, *, axis, coord_name, long_name, standard_name, units):
        if standard_name == "forecast_period":
            return StepCoordinate(c)

        if long_name == "time elapsed since the start of the forecast":
            return StepCoordinate(c)

    def _is_level(self, c, *, axis, coord_name, long_name, standard_name, units):
        if standard_name == "atmosphere_hybrid_sigma_pressure_coordinate":
            return LevelCoordinate(c, "ml")

        if long_name == "height" and units == "m":
            return LevelCoordinate(c, "height")

        if standard_name == "air_pressure" and units == "hPa":
            return LevelCoordinate(c, "pl")

        if coord_name == "level":
            return LevelCoordinate(c, "pl")

        if coord_name == "vertical" and units == "hPa":
            return LevelCoordinate(c, "pl")

        if standard_name == "depth":
            return LevelCoordinate(c, "depth")

    def grid(self, coordinates):
        lat = [c for c in coordinates if c.is_lat]
        lon = [c for c in coordinates if c.is_lon]

        if len(lat) != 1:
            raise NotImplementedError(f"Expected 1 latitude coordinate, got {len(lat)}")

        if len(lon) != 1:
            raise NotImplementedError(f"Expected 1 longitude coordinate, got {len(lon)}")

        lat = lat[0]
        lon = lon[0]

        if (lat.name, lon.name) in self._cache:
            return self._cache[(lat.name, lon.name)]

        assert len(lat.variable.shape) == len(lon.variable.shape), (lat.variable.shape, lon.variable.shape)
        if len(lat.variable.shape) == 1:
            grid = MeshedGrid(lat, lon)
        else:
            grid = UnstructuredGrid(lat, lon)

        self._cache[(lat.name, lon.name)] = grid
        return grid
