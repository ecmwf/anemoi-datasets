# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
import math
import textwrap
from functools import cached_property

import numpy as np
from earthkit.data.core.fieldlist import Field
from earthkit.data.core.fieldlist import FieldList
from earthkit.data.core.fieldlist import MultiFieldList
from earthkit.data.core.geography import Geography
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.utils.dates import to_datetime
from earthkit.data.utils.projections import Projection

LOG = logging.getLogger(__name__)


def _is_scalar(variable):
    shape = variable.shape
    if shape == (1,):
        return True
    if len(shape) == 0:
        return True
    return False


def _extract_single_value(variable):
    shape = variable.shape
    if np.issubdtype(variable.values.dtype, np.datetime64):
        if len(shape) == 0:
            return to_datetime(variable.values)  # Convert to python datetime
        assert False, (shape, variable.values)

    if shape == (1,):
        return variable.values[0]

    if len(shape) == 0:
        return variable.values.item()

    assert False, (shape, variable.values)


class Coordinate:
    is_grid = False
    is_dim = True
    is_lat = False
    is_lon = False

    def __init__(self, variable):
        self.variable = variable
        self.scalar = _is_scalar(variable)
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


class LevelCoordinate(Coordinate):

    def __init__(self, variable, levtype):
        super().__init__(variable)
        self.levtype = levtype
        self.kwargs = {"levtype": levtype}


class EnsembleCoordinate(Coordinate):
    pass


class OtherCoordinate(Coordinate):
    pass


class EmptyFieldList:
    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError(i)


class XArrayFieldGeography(Geography):
    def __init__(self, field):
        self._field = field

    def _unique_grid_id(self):
        raise NotImplementedError()

    def bounding_box(self):
        raise NotImplementedError()
        # return BoundingBox(north=self.north, south=self.south, east=self.east, west=self.west)

    def gridspec(self):
        raise NotImplementedError()

    def latitudes(self, dtype=None):
        result = self._field.grid.lat.variable.values
        if dtype is not None:
            return result.astype(dtype)
        return result

    def longitudes(self, dtype=None):
        result = self._field.grid.lon.variable.values
        if dtype is not None:
            return result.astype(dtype)
        return result

    def resolution(self):
        # TODO: implement resolution
        return None

    @property
    def mars_grid(self):
        # TODO: implement mars_grid
        return None

    @property
    def mars_area(self):
        # TODO: code me
        # return [self.north, self.west, self.south, self.east]
        return None

    def x(self, dtype=None):
        raise NotImplementedError()

    def y(self, dtype=None):
        raise NotImplementedError()

    def shape(self):
        return self._field.shape

    def projection(self):
        return Projection.from_cf_grid_mapping(**self._field.grid_mapping)


class XArrayMetadata(RawMetadata):
    LS_KEYS = ["variable", "level", "valid_datetime", "units"]
    NAMESPACES = ["default", "mars"]
    MARS_KEYS = ["param", "step", "levelist", "levtype", "number", "date", "time"]

    def __init__(self, field):
        self._field = field
        md = field._md.copy()

        time = to_datetime(md.pop("time"))
        base = to_datetime(self._base_datetime())

        step = (time - base).total_seconds() // 3600
        assert step >= 0
        assert step == int(step)

        md["step"] = int(step)
        md["date"] = base.strftime("%Y%m%d")
        md["time"] = base.strftime("%H%M")

        self._time = time

        super().__init__(md)

    @cached_property
    def geography(self):
        return XArrayFieldGeography(self._field)

    def as_namespace(self, namespace=None):
        if not isinstance(namespace, str) and namespace is not None:
            raise TypeError("namespace must be a str or None")

        if namespace == "default" or namespace == "" or namespace is None:
            assert False
            return dict(self)
        elif namespace == "mars":
            return self._as_mars()

    def _as_mars(self):
        return dict(
            param=self["variable"],
            step=self["step"],
            levelist=self["level"],
            levtype=self["levtype"],
            number=self["number"],
            date=self["date"],
            time=self["time"],
        )

    def _base_datetime(self):
        return self._field.forecast_reference_time

    def _valid_datetime(self):
        return self._time

    def _get(self, key, **kwargs):

        if key.startswith("mars."):
            key = key[5:]
            if key not in self.MARS_KEYS:
                if kwargs.get("raise_on_missing", False):
                    raise KeyError(f"Invalid key '{key}' in namespace='mars'")
                else:
                    return kwargs.get("default", None)

        _key_name = {"param": "variable", "levelist": "level"}

        return super()._get(_key_name.get(key, key), **kwargs)


class XArrayField(Field):

    def __init__(self, owner, selection):
        super().__init__(owner)
        self.owner = owner
        self.selection = selection
        self._md = owner._metadata.copy()

        for coord_name, coord_value in self.selection.coords.items():
            if coord_name in selection.dims:
                continue

            if _is_scalar(coord_value):
                self._md[coord_name] = _extract_single_value(coord_value)

    def to_numpy(self, flatten=False, dtype=None):
        assert dtype is None
        if flatten:
            return self.selection.values.flatten()
        return self.selection.values

    def _make_metadata(self):
        return XArrayMetadata(self)

    def grid_points(self):
        return self.owner.grid_points()

    @property
    def resolution(self):
        return None

    @property
    def shape(self):
        return self.selection.shape

    @property
    def grid_mapping(self):
        return self.owner.grid_mapping

    @property
    def forecast_reference_time(self):
        return self.owner.forecast_reference_time

    def __repr__(self):
        return textwrap.shorten("Field[%s]" % (self._metadata), width=80, placeholder="...")


class Variable:
    def __init__(self, *, ds, var, coordinates, grid, forecast_reference_time, metadata):
        self.ds = ds
        self.var = var
        self.forecast_reference_time = forecast_reference_time
        self.grid = grid
        self.coordinates = coordinates

        self._metadata = metadata.copy()
        self._metadata.update(var.attrs)
        self._metadata.update({"variable": var.name})

        self._metadata.setdefault("level", None)
        self._metadata.setdefault("number", 0)
        self._metadata.setdefault("levtype", "sfc")

        self.shape = tuple(len(c.variable) for c in coordinates if c.is_dim and not c.scalar and not c.is_grid)
        self.names = {c.variable.name: c for c in coordinates if c.is_dim and not c.scalar and not c.is_grid}
        self.by_name = {c.variable.name: c for c in coordinates}

        self.length = math.prod(self.shape)

    @property
    def grid_mapping(self):
        grid_mapping = self.var.attrs.get("grid_mapping", None)
        if grid_mapping is None:
            return None
        return self.ds[grid_mapping].attrs

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
        return XArrayField(self, self.var.isel(kwargs))

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
            ds=self.ds,
            var=self.var.isel({k: i}),
            coordinates=coordinates,
            grid=self.grid,
            forecast_reference_time=self.forecast_reference_time,
            metadata=metadata,
        )

        return variable.sel(**kwargs)

    def match(self, **kwargs):
        for k, v in list(kwargs.items()):

            if not isinstance(v, list):
                v = [v]

            name = "variable" if k == "param" else k

            if name in self._metadata:
                if self._metadata[name] not in v:
                    return False, None

            kwargs.pop(k)

        return True, kwargs


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


class XarrayFieldList(FieldList):
    def __init__(self, ds, variables):
        self.ds = ds
        self.variables = variables.copy()
        self.total_length = sum(v.length for v in variables)

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
    def from_xarray(cls, ds):
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

        forecast_reference_time = None
        # Special variables
        for name in ds.data_vars:
            if name in skip:
                continue

            v = ds[name]
            if v.attrs.get("standard_name", "").lower() == "forecast_reference_time":
                forecast_reference_time = _extract_single_value(v)
                continue

        # Select only geographical variables
        for name in ds.data_vars:

            if name in skip:
                continue

            v = ds[name]
            coordinates = []

            for coord in v.coords:

                c = guess.guess(ds[coord], coord)
                assert c, f"Could not guess coordinate for {coord}"
                if coord not in v.dims:
                    c.is_dim = False
                coordinates.append(c)

            grid_coords = sum(1 for c in coordinates if c.is_grid and c.is_dim)
            assert grid_coords <= 2

            if grid_coords < 2:
                continue

            variables.append(
                Variable(
                    ds=ds,
                    var=v,
                    coordinates=coordinates,
                    grid=guess.grid(coordinates),
                    forecast_reference_time=forecast_reference_time,
                    metadata={},
                )
            )

        return cls(ds, variables)

    def sel(self, **kwargs):

        variables = []
        for v in self.variables:
            match, rest = v.match(**kwargs)
            if match:
                v = v.sel(**rest)
                if v is not None:
                    variables.append(v)

        if not variables:
            return EmptyFieldList()

        return self.__class__(self.ds, variables)


def execute(context, dates, dataset, options, *args, **kwargs):
    import xarray as xr

    context.trace("🌐", dataset, options)

    if isinstance(dataset, str) and ".zarr" in dataset:
        data = xr.open_zarr(dataset, **options)
    else:
        data = xr.open_dataset(dataset, **options)

    fs = XarrayFieldList.from_xarray(data)
    return MultiFieldList([fs.sel(time=date, **kwargs) for date in dates])
