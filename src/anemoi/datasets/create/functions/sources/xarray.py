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

import numpy as np
from earthkit.data.core.fieldlist import FieldList
from earthkit.data.core.fieldlist import MultiFieldList

LOG = logging.getLogger(__name__)

GEOGRAPHIC_COORDS = {
    "x": ["x", "projection_x_coordinate", "lon", "longitude"],
    "y": ["y", "projection_y_coordinate", "lat", "latitude"],
}


class Coordinate:
    def __init__(self, variable):
        self.variable = variable
        self.scalar = variable.shape == tuple()
        print(self)

    def __len__(self):
        return 1 if self.scalar else len(self.variable)

    def __repr__(self):
        return "%s[name=%s,values=%s]" % (
            self.__class__.__name__,
            self.variable.name,
            self.variable.values if self.scalar else len(self),
        )

    def singleton(self, i):
        return self.__class__(self.variable.isel({self.variable.dims[0]: i}))

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


class TimeCoordinate(Coordinate):

    def index(self, time):
        return super().index(np.datetime64(time))


class LevelCoordinate(Coordinate):
    pass


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
            if np.issubdtype(coord_value.dtype, np.datetime64):
                # self._metadata[coord_name] = coord_value.values.astype(object)
                self._metadata[coord_name] = str(coord_value.values).split(".")[0]
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
        return "Field[%s]" % (self._metadata)


class Variable:
    def __init__(self, ds, var, coordinates, lat, lon, metadata={}):
        self.ds = ds
        self.var = var
        self.lat = lat
        self.lon = lon
        self._metadata = metadata.copy()
        self._metadata.update(var.attrs)
        self._metadata.update({"variable": var.name})

        self.coordinates = coordinates
        self.shape = tuple(len(c.variable) for c in coordinates if not c.scalar)
        self.names = {c.variable.name: c for c in coordinates if not c.scalar}
        self.by_name = {c.variable.name: c for c in coordinates}

        self.length = math.prod(self.shape)

    def grid_points(self):
        x, y = np.meshgrid(self.ds[self.lat].values, self.ds[self.lon].values)
        return x.flatten(), y.flatten()

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
        variable = Variable(self.ds, self.var.isel({k: i}), coordinates, self.lat, self.lon, metadata)

        return variable.sel(**kwargs)


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
        coordinates_cache = {}

        skip = set()

        def _add_coordinates(name, coordinates, coordinate):
            coordinates.append(coordinate)
            coordinates_cache[name] = coordinate

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
            has_lat = None
            has_lon = None

            if name in skip:
                continue

            v = ds[name]

            coordinates = []

            # self.log.info('Scanning file: %s var=%s coords=%s', self.path, name, v.coords)

            # info = [value for value in v.coords if value not in v.dims]
            non_dim_coords = {}
            for coord in v.coords:
                if coord not in v.dims:
                    non_dim_coords[coord] = ds[coord].values
                    continue

                if coord in coordinates_cache:
                    coordinates.append(coordinates_cache[coord])
                    continue

                c = ds[coord]

                # self.log.info("COORD %s %s %s %s", coord, type(coord), hasattr(c, 'calendar'), c)

                standard_name = getattr(c, "standard_name", "")
                axis = getattr(c, "axis", "")
                long_name = getattr(c, "long_name", "")
                coord_name = getattr(c, "name", "")

                # LOG.debug(f"{standard_name=} {long_name=} {axis=} {coord_name}")
                use = False

                if (
                    standard_name.lower() in GEOGRAPHIC_COORDS["x"]
                    or (long_name == "longitude")
                    or (axis == "X")
                    or coord_name.lower() in GEOGRAPHIC_COORDS["x"]
                ):
                    has_lon = coord
                    use = True

                if (
                    standard_name.lower() in GEOGRAPHIC_COORDS["y"]
                    or (long_name == "latitude")
                    or (axis == "Y")
                    or coord_name.lower() in GEOGRAPHIC_COORDS["y"]
                ):
                    has_lat = coord
                    use = True

                # Of course, not every one sets the standard_name
                if (
                    standard_name in ["time", "forecast_reference_time"]
                    or long_name in ["time"]
                    or coord_name.lower() in ["time"]
                    or axis == "T"
                ):
                    # we might not be able to convert time to datetime
                    try:
                        _add_coordinates(coord, coordinates, TimeCoordinate(c))
                        use = True
                    except ValueError:
                        break

                # TODO: Support other level types
                if (
                    standard_name
                    in [
                        "air_pressure",
                        "model_level_number",
                        "altitude",
                    ]
                    or long_name in ["pressure_level"]
                    or coord_name in ["level"]
                ):  # or axis == 'Z':
                    _add_coordinates(coord, coordinates, LevelCoordinate(c))
                    use = True

                if axis in ("X", "Y"):
                    use = True

                if not use:
                    _add_coordinates(coord, coordinates, OtherCoordinate(c))

            if not (has_lat and has_lon):
                # self.log.info("NetCDFReader: skip %s (Not a 2 field)", name)
                continue

            variables.append(Variable(ds, v, coordinates, has_lat, has_lon))

        return cls(ds, variables)

    def sel(self, **kwargs):
        variables = kwargs.pop("variables", kwargs.pop("param", None))
        if variables is not None:
            variables = [v for v in self.variables if v.var.name in variables]
            if not variables:
                return EmptyFieldList()
            return self.__class__(self.ds, variables).sel(**kwargs)

        variables = [v.sel(**kwargs) for v in self.variables]
        variables = [v for v in variables if v is not None]
        if not variables:
            return EmptyFieldList()

        return self.__class__(self.ds, variables)


def execute(context, dates, dataset, options, *args, **kwargs):
    import xarray as xr

    data = xr.open_zarr(dataset, **options)
    fs = XarrayFieldList.from_xarray(data)
    return MultiFieldList([fs.sel(time=date, **kwargs) for date in dates])
