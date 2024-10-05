# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging

from .coordinates import DateCoordinate
from .coordinates import EnsembleCoordinate
from .coordinates import LatitudeCoordinate
from .coordinates import LevelCoordinate
from .coordinates import LongitudeCoordinate
from .coordinates import ScalarCoordinate
from .coordinates import StepCoordinate
from .coordinates import TimeCoordinate
from .coordinates import UnsupportedCoordinate
from .coordinates import XCoordinate
from .coordinates import YCoordinate
from .coordinates import is_scalar
from .grid import MeshedGrid
from .grid import MeshProjectionGrid
from .grid import UnstructuredGrid
from .grid import UnstructuredProjectionGrid

LOG = logging.getLogger(__name__)


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

        d = self._is_number(
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

        LOG.warning(
            f"Coordinate {coord} not supported\n{axis=}, {name=},"
            f" {long_name=}, {standard_name=}, units\n\n{c}\n\n{type(c.values)} {c.shape}"
        )

        return UnsupportedCoordinate(c)

    def grid(self, coordinates, variable):
        lat = [c for c in coordinates if c.is_lat]
        lon = [c for c in coordinates if c.is_lon]

        if len(lat) == 1 and len(lon) == 1:
            return self._lat_lon_provided(lat, lon, variable)

        x = [c for c in coordinates if c.is_x]
        y = [c for c in coordinates if c.is_y]

        if len(x) == 1 and len(y) == 1:
            return self._x_y_provided(x, y, variable)

        raise NotImplementedError(f"Cannot establish grid {coordinates}")

    def _check_dims(self, variable, x_or_lon, y_or_lat):

        x_dims = set(x_or_lon.variable.dims)
        y_dims = set(y_or_lat.variable.dims)
        variable_dims = set(variable.dims)

        if not (x_dims <= variable_dims) or not (y_dims <= variable_dims):
            raise ValueError(
                f"Dimensions do not match {variable.name}{variable.dims} !="
                f" {x_or_lon.name}{x_or_lon.variable.dims} and {y_or_lat.name}{y_or_lat.variable.dims}"
            )

        variable_dims = tuple(v for v in variable.dims if v in (x_dims | y_dims))
        if x_dims == y_dims:
            # It's unstructured
            return variable_dims, True

        if len(x_dims) == 1 and len(y_dims) == 1:
            # It's a mesh
            return variable_dims, False

        raise ValueError(
            f"Cannot establish grid for {variable.name}{variable.dims},"
            f" {x_or_lon.name}{x_or_lon.variable.dims},"
            f" {y_or_lat.name}{y_or_lat.variable.dims}"
        )

    def _lat_lon_provided(self, lat, lon, variable):
        lat = lat[0]
        lon = lon[0]

        dim_vars, unstructured = self._check_dims(variable, lon, lat)

        if (lat.name, lon.name, dim_vars) in self._cache:
            return self._cache[(lat.name, lon.name, dim_vars)]

        if unstructured:
            grid = UnstructuredGrid(lat, lon, dim_vars)
        else:
            grid = MeshedGrid(lat, lon, dim_vars)

        self._cache[(lat.name, lon.name, dim_vars)] = grid
        return grid

    def _x_y_provided(self, x, y, variable):
        x = x[0]
        y = y[0]

        dim_vars, unstructured = self._check_dims(variable, x, y)

        if (x.name, y.name, dim_vars) in self._cache:
            return self._cache[(x.name, y.name, dim_vars)]

        grid_mapping = variable.attrs.get("grid_mapping", None)
        if grid_mapping is not None:
            print(f"grid_mapping: {grid_mapping}")
            print(self.ds[grid_mapping])

        if grid_mapping is None:
            LOG.warning(f"No 'grid_mapping' attribute provided for '{variable.name}'")
            LOG.warning("Trying to guess...")

            PROBE = {
                "prime_meridian_name",
                "reference_ellipsoid_name",
                "crs_wkt",
                "horizontal_datum_name",
                "semi_major_axis",
                "spatial_ref",
                "inverse_flattening",
                "semi_minor_axis",
                "geographic_crs_name",
                "GeoTransform",
                "grid_mapping_name",
                "longitude_of_prime_meridian",
            }
            candidate = None
            for v in self.ds.variables:
                var = self.ds[v]
                if not is_scalar(var):
                    continue

                if PROBE.intersection(var.attrs.keys()):
                    if candidate:
                        raise ValueError(f"Multiple candidates for 'grid_mapping': {candidate} and {v}")
                    candidate = v

            if candidate:
                LOG.warning(f"Using '{candidate}' as 'grid_mapping'")
                grid_mapping = candidate
            else:
                LOG.warning("Could not fine a candidate for 'grid_mapping'")

        if grid_mapping is None:
            if "crs" in self.ds[variable].attrs:
                grid_mapping = self.ds[variable].attrs["crs"]
                LOG.warning(f"Using CRS {grid_mapping} from variable '{variable.name}' attributes")

        if grid_mapping is None:
            if "crs" in self.ds.attrs:
                grid_mapping = self.ds.attrs["crs"]
                LOG.warning(f"Using CRS {grid_mapping} from global attributes")

        grid = None
        if grid_mapping is not None:

            grid_mapping = dict(self.ds[grid_mapping].attrs)

            if unstructured:
                grid = UnstructuredProjectionGrid(x, y, grid_mapping)
            else:
                grid = MeshProjectionGrid(x, y, grid_mapping)

        if grid is not None:
            self._cache[(x.name, y.name, dim_vars)] = grid
            return grid

        LOG.error("Could not fine a candidate for 'grid_mapping'")
        raise NotImplementedError(f"Unstructured grid {x.name} {y.name}")


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

        # That is the output of `cfgrib` for forecasts
        if name == "time" and standard_name != "forecast_reference_time":
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

        if name == "vertical" and units == "hPa":
            return LevelCoordinate(c, "pl")

    def _is_number(self, c, *, axis, name, long_name, standard_name, units):
        if name in ("realization", "number"):
            return EnsembleCoordinate(c)


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

    def _is_number(self, c, *, axis, name, long_name, standard_name, units):
        if self._match(c, "number", locals()):
            return DateCoordinate(c)
