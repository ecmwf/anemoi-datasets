# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import json
import logging
import math
import textwrap

import numpy as np
from earthkit.data.core.fieldlist import FieldList
from earthkit.data.core.fieldlist import MultiFieldList

LOG = logging.getLogger(__name__)


class Coordinate:
    is_grid = False
    is_dim = True
    is_lat = False
    is_lon = False

    def __init__(self, variable):
        self.variable = variable
        self.scalar = variable.shape == tuple()
        self.kwargs = {}
        # print(self)

    def __len__(self):
        return 1 if self.scalar else len(self.variable)

    def __repr__(self):
        return "%s[name=%s,values=%s]" % (
            self.__class__.__name__,
            self.variable.name,
            self.variable.values if self.scalar else len(self),
        )

    def singleton(self, i):
        return self.__class__(self.variable.isel({self.variable.dims[0]: i}), **self.kwargs)

    def index(self, value):

        values = self.variable.values

        # Assume the array is sorted
        index = np.searchsorted(values, value)
        if index < len(values) and values[index] == value:
            return index

        # If not found, we need to check if the value is in the array

        index = np.where(values == value)[0]
        if len(index) > 0:
            return index[0]

        return None

    @property
    def name(self):
        return self.variable.name


class TimeCoordinate(Coordinate):

    def index(self, time):
        return super().index(np.datetime64(time))


class StepCoordinate(Coordinate):
    pass
    # def index(self, time):
    #     return super().index(np.datetime64(time))


class LevelCoordinate(Coordinate):

    def __init__(self, variable, levtype):
        super().__init__(variable)
        self.levtype = levtype
        self.kwargs = {"levtype": levtype}


class EnsembleCoordinate(Coordinate):
    # TODO: Implement
    pass


class OtherCoordinate(Coordinate):
    pass


class EmptyFieldList:
    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError(i)


class Field:
    __slots__ = ["owner", "selection", "_metadata"]

    def __init__(self, owner, selection):
        self.owner = owner
        self.selection = selection
        self._metadata = owner._metadata.copy()

        for coord_name, coord_value in self.selection.coords.items():
            if coord_name in selection.dims:
                continue
            # print(coord_name, coord_value)
            if np.issubdtype(coord_value.dtype, np.datetime64):
                # self._metadata[coord_name] = coord_value.values.astype(object)
                self._metadata[coord_name] = str(coord_value.values).split(".")[0]
            else:
                if isinstance(coord_value.values, np.ndarray):
                    self._metadata[coord_name] = coord_value.values.tolist()
                else:
                    self._metadata[coord_name] = coord_value.values.item()

    def to_numpy(self, flatten=False, dtype=None):
        assert dtype is None
        if flatten:
            return self.selection.values.flatten()
        return self.selection.values

    def metadata(self, key, default=None):
        if "valid_datetime" == key:
            key = "time"
        if "param" == key:
            key = "variable"
        return self._metadata.get(key, default)

    def grid_points(self):
        return self.owner.grid_points()

    @property
    def resolution(self):
        return None

    @property
    def shape(self):
        return self.selection.shape

    def __repr__(self):
        return textwrap.shorten("Field[%s]" % (self._metadata), width=80, placeholder="...")


class Variable:
    def __init__(self, ds, var, coordinates, grid, mars_names, metadata_handler, metadata):
        self.ds = ds
        self.var = var
        self.mars_names = mars_names
        self.metadata_handler = metadata_handler
        self._metadata = metadata.copy()
        self._metadata.update(var.attrs)
        self._metadata.update({"variable": var.name})
        self.grid = grid

        self.coordinates = coordinates
        self.shape = tuple(len(c.variable) for c in coordinates if c.is_dim and not c.scalar and not c.is_grid)
        self.names = {c.variable.name: c for c in coordinates if c.is_dim and not c.scalar and not c.is_grid}
        self.by_name = {c.variable.name: c for c in coordinates}

        self.length = math.prod(self.shape)

    def grid_points(self):
        return self.grid.grid_points()

    def __repr__(self):
        return "Variable[name=%s,coordinates=%s,lat=%s,lon=%s,metadata=%s]" % (
            self.var.name,
            self.coordinates,
            self.lat,
            self.lon,
            self._metadata,
        )

    def __getitem__(self, i):
        coords = np.unravel_index(i, self.shape)
        kwargs = {k: v for k, v in zip(self.names, coords)}
        return Field(self, self.var.isel(kwargs))

    def sel(self, **kwargs):

        if not kwargs:
            return self

        k, v = kwargs.popitem()
        c = self.by_name.get(k)
        if c is None:
            return None

        i = c.index(v)
        if i is None:
            return None

        coordinates = [x.singleton(i) if c is x else x for x in self.coordinates]

        metadata = self._metadata.copy()
        metadata.update({k: v})

        variable = Variable(
            self.ds,
            self.var.isel({k: i}),
            coordinates,
            self.grid,
            self.mars_names,
            self.metadata_handler,
            metadata,
        )

        return variable.sel(**kwargs)


class LongitudeCoordinate(Coordinate):
    is_grid = True
    is_lon = True


class LatitudeCoordinate(Coordinate):
    is_grid = True
    is_lat = True


class XCoordinate(Coordinate):
    is_grid = True


class YCoordinate(Coordinate):
    is_grid = True


class ScalarCoordinate(Coordinate):
    pass


class Grid:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon


class MeshedGrid(Grid):
    _cache = None

    def grid_points(self):
        if self._cache is not None:
            return self._cache
        lat = self.lat.variable.values
        lon = self.lon.variable.values

        lat, lon = np.meshgrid(lat, lon)
        self._cache = (lat.flatten(), lon.flatten())
        return self._cache


class UnstructuredGrid(Grid):
    def grid_points(self):
        lat = self.lat.variable.values.flatten()
        lon = self.lon.variable.values.flatten()
        return lat, lon


class CoordinateGuesser:

    def __init__(self, ds):
        self.ds = ds
        self._cache = {}

    def guess(self, c, coord):
        if coord not in self._cache:
            self._cache[coord] = self._guess(c, coord)
        return self._cache[coord]

    def _guess(self, c, coord):
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

    def _is_longitude(self, c, axis, coord_name, long_name, standard_name, units):
        if standard_name == "longitude":
            return LongitudeCoordinate(c)

        if long_name == "longitude" and units == "degrees_east":
            return LongitudeCoordinate(c)

    def _is_latitude(self, c, axis, coord_name, long_name, standard_name, units):
        if standard_name == "latitude":
            return LatitudeCoordinate(c)

        if long_name == "latitude" and units == "degrees_north":
            return LatitudeCoordinate(c)

    def _is_x(self, c, axis, coord_name, long_name, standard_name, units):
        if standard_name == "projection_x_coordinate":
            return XCoordinate(c)

    def _is_y(self, c, axis, coord_name, long_name, standard_name, units):
        if standard_name == "projection_y_coordinate":
            return YCoordinate(c)

    def _is_time(self, c, axis, coord_name, long_name, standard_name, units):
        if standard_name == "time":
            return TimeCoordinate(c)

        if coord_name == "time":
            return TimeCoordinate(c)

    def _is_step(self, c, axis, coord_name, long_name, standard_name, units):
        if standard_name == "forecast_period":
            return StepCoordinate(c)

        if long_name == "time elapsed since the start of the forecast":
            return StepCoordinate(c)

    def _is_level(self, c, axis, coord_name, long_name, standard_name, units):
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


class XarrayFieldList(FieldList):
    def __init__(self, ds, variables, metadata_handler):
        self.ds = ds
        self.variables = variables.copy()
        self.total_length = sum(v.length for v in variables)
        self.metadata_handler = metadata_handler

    def __len__(self):
        return self.total_length

    def __getitem__(self, i):
        k = i

        if i < 0:
            i = self.total_length + i

        for v in self.variables:
            if i < v.length:
                return v[i]
            i -= v.length

        raise IndexError(k)

    @classmethod
    def from_xarray(cls, ds, metadata_handler=None):
        variables = []
        guess = CoordinateGuesser(ds)

        skip = set()

        def _skip_attr(v, attr_name):
            attr_val = getattr(v, attr_name, "")
            if isinstance(attr_val, str):
                skip.update(attr_val.split(" "))

        for name in ds.data_vars:
            v = ds[name]
            _skip_attr(v, "coordinates")
            _skip_attr(v, "bounds")
            _skip_attr(v, "grid_mapping")

        for name in ds.data_vars:
            # Select only geographical variables
            # mars_names = {"levtype": "sfc"}

            if name in skip:
                continue

            v = ds[name]
            # print(v)

            coordinates = []

            # self.log.info('Scanning file: %s var=%s coords=%s', self.path, name, v.coords)

            # info = [value for value in v.coords if value not in v.dims]
            # non_dim_coords = []
            for coord in v.coords:

                c = guess.guess(ds[coord], coord)
                assert c, f"Could not guess coordinate for {coord}"
                if coord not in v.dims:
                    c.is_dim = False
                coordinates.append(c)

            grid_coords = sum(1 for c in coordinates if c.is_grid and c.is_dim)
            assert grid_coords <= 2

            if grid_coords < 2:
                # self.log.info("NetCDFReader: skip %s (Not a 2 field)", name)
                continue

            variables.append(
                Variable(
                    ds,
                    v,
                    coordinates,
                    guess.grid(coordinates),
                    {},
                    metadata_handler,
                    {},
                )
            )

        return cls(ds, variables, metadata_handler)

    def sel(self, **kwargs):
        variables = kwargs.pop("variables", kwargs.pop("param", None))
        if variables is not None:
            variables = [v for v in self.variables if v.var.name in variables]
            if not variables:
                return EmptyFieldList()
            # print("++++++", len(variables))
            return self.__class__(self.ds, variables, self.metadata_handler).sel(**kwargs)

        variables = [v.sel(**kwargs) for v in self.variables]
        variables = [v for v in variables if v is not None]
        if not variables:
            return EmptyFieldList()

        return self.__class__(self.ds, variables, self.metadata_handler)


def mars_naming(name, metadata, default=None):
    return metadata.get(name, default)


def execute(context, dates, dataset, options, *args, **kwargs):
    import xarray as xr

    print(json.dumps([dataset, options], indent=4))
    if isinstance(dataset, str) and ".zarr" in dataset:
        data = xr.open_zarr(dataset, **options)
    else:
        data = xr.open_dataset(dataset, **options)

    fs = XarrayFieldList.from_xarray(data, mars_naming)
    return MultiFieldList([fs.sel(time=date, **kwargs) for date in dates])
