# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from .coordinates import DateCoordinate
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

    def __init__(self, ds):
        self.ds = ds
        self._cache = {}

    @classmethod
    def from_flavour(cls, ds, flavour):
        if flavour is None:
            return DefaultCoordinateGuesser(ds)
        else:
            return FlavourCoordinateGuesser(ds, flavour)

    def guess(self, c, coord):
        if coord not in self._cache:
            self._cache[coord] = self._guess(c, coord)
        return self._cache[coord]

    def _guess(self, c, coord):

        name = c.name
        standard_name = getattr(c, "standard_name", "").lower()
        axis = getattr(c, "axis", "")
        long_name = getattr(c, "long_name", "").lower()
        units = getattr(c, "units", "")

        d = self._is_longitude(
            c,
            axis=axis,
            name=name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_latitude(
            c,
            axis=axis,
            name=name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_x(
            c,
            axis=axis,
            name=name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_y(
            c,
            axis=axis,
            name=name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_time(
            c,
            axis=axis,
            name=name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_step(
            c,
            axis=axis,
            name=name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_date(
            c,
            axis=axis,
            name=name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        d = self._is_level(
            c,
            axis=axis,
            name=name,
            long_name=long_name,
            standard_name=standard_name,
            units=units,
        )
        if d is not None:
            return d

        if c.shape in ((1,), tuple()):
            return ScalarCoordinate(c)

        raise NotImplementedError(
            f"Coordinate {coord} not supported\n{axis=}, {name=},"
            f" {long_name=}, {standard_name=}, units\n\n{c}\n\n{type(c.values)} {c.shape}"
        )

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


class DefaultCoordinateGuesser(CoordinateGuesser):
    def __init__(self, ds):
        super().__init__(ds)

    def _is_longitude(self, c, *, axis, name, long_name, standard_name, units):
        if standard_name == "longitude":
            return LongitudeCoordinate(c)

        if long_name == "longitude" and units == "degrees_east":
            return LongitudeCoordinate(c)

        if name == "longitude":  # WeatherBench
            return LongitudeCoordinate(c)

    def _is_latitude(self, c, *, axis, name, long_name, standard_name, units):
        if standard_name == "latitude":
            return LatitudeCoordinate(c)

        if long_name == "latitude" and units == "degrees_north":
            return LatitudeCoordinate(c)

        if name == "latitude":  # WeatherBench
            return LatitudeCoordinate(c)

    def _is_x(self, c, *, axis, name, long_name, standard_name, units):
        if standard_name == "projection_x_coordinate":
            return XCoordinate(c)

        if name == "x":
            return XCoordinate(c)

    def _is_y(self, c, *, axis, name, long_name, standard_name, units):
        if standard_name == "projection_y_coordinate":
            return YCoordinate(c)

        if name == "y":
            return YCoordinate(c)

    def _is_time(self, c, *, axis, name, long_name, standard_name, units):
        if standard_name == "time":
            return TimeCoordinate(c)

        if name == "time":
            return TimeCoordinate(c)

    def _is_date(self, c, *, axis, name, long_name, standard_name, units):
        if standard_name == "forecast_reference_time":
            return DateCoordinate(c)
        if name == "forecast_reference_time":
            return DateCoordinate(c)

    def _is_step(self, c, *, axis, name, long_name, standard_name, units):
        if standard_name == "forecast_period":
            return StepCoordinate(c)

        if long_name == "time elapsed since the start of the forecast":
            return StepCoordinate(c)

        if name == "prediction_timedelta":  # WeatherBench
            return StepCoordinate(c)

    def _is_level(self, c, *, axis, name, long_name, standard_name, units):
        if standard_name == "atmosphere_hybrid_sigma_pressure_coordinate":
            return LevelCoordinate(c, "ml")

        if long_name == "height" and units == "m":
            return LevelCoordinate(c, "height")

        if standard_name == "air_pressure" and units == "hPa":
            return LevelCoordinate(c, "pl")

        if name == "level":
            return LevelCoordinate(c, "pl")

        if name == "vertical" and units == "hPa":
            return LevelCoordinate(c, "pl")

        if standard_name == "depth":
            return LevelCoordinate(c, "depth")

        if name == "pressure":
            return LevelCoordinate(c, "pl")


class FlavourCoordinateGuesser(CoordinateGuesser):
    def __init__(self, ds, flavour):
        super().__init__(ds)
        self.flavour = flavour

    def _match(self, c, key, values):

        if key not in self.flavour["rules"]:
            return None

        rules = self.flavour["rules"][key]

        if not isinstance(rules, list):
            rules = [rules]

        for rule in rules:
            ok = True
            for k, v in rule.items():
                if isinstance(v, str) and values.get(k) != v:
                    ok = False
            if ok:
                return rule

        return None

    def _is_longitude(self, c, *, axis, name, long_name, standard_name, units):
        if self._match(c, "longitude", locals()):
            return LongitudeCoordinate(c)

    def _is_latitude(self, c, *, axis, name, long_name, standard_name, units):
        if self._match(c, "latitude", locals()):
            return LatitudeCoordinate(c)

    def _is_x(self, c, *, axis, name, long_name, standard_name, units):
        if self._match(c, "x", locals()):
            return XCoordinate(c)

    def _is_y(self, c, *, axis, name, long_name, standard_name, units):
        if self._match(c, "y", locals()):
            return YCoordinate(c)

    def _is_time(self, c, *, axis, name, long_name, standard_name, units):
        if self._match(c, "time", locals()):
            return TimeCoordinate(c)

    def _is_step(self, c, *, axis, name, long_name, standard_name, units):
        if self._match(c, "step", locals()):
            return StepCoordinate(c)

    def _is_date(self, c, *, axis, name, long_name, standard_name, units):
        if self._match(c, "date", locals()):
            return DateCoordinate(c)

    def _is_level(self, c, *, axis, name, long_name, standard_name, units):

        rule = self._match(c, "level", locals())
        if rule:
            # assert False, rule
            return LevelCoordinate(
                c,
                self._levtype(
                    c,
                    axis=axis,
                    name=name,
                    long_name=long_name,
                    standard_name=standard_name,
                    units=units,
                ),
            )

    def _levtype(self, c, *, axis, name, long_name, standard_name, units):
        if "levtype" in self.flavour:
            return self.flavour["levtype"]

        raise NotImplementedError(f"levtype for {c=}")
