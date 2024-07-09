# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import datetime
import logging
import math
from functools import cached_property

import numpy as np
from earthkit.data.core.fieldlist import Field
from earthkit.data.core.fieldlist import FieldList
from earthkit.data.core.fieldlist import MultiFieldList

from .coordinates import _extract_single_value
from .coordinates import _is_scalar
from .flavour import CoordinateGuesser
from .metadata import XArrayMetadata

LOG = logging.getLogger(__name__)


class EmptyFieldList:
    def __len__(self):
        return 0

    def __getitem__(self, i):
        raise IndexError(i)

    def __repr__(self) -> str:
        return "EmptyFieldList()"


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
    def latitudes(self):
        return self.owner.latitudes

    @property
    def longitudes(self):
        return self.owner.longitudes

    @property
    def forecast_reference_time(self):
        return self.owner.forecast_reference_time

    def __repr__(self):
        return repr(self._metadata)


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

    def __len__(self):
        return self.length

    @property
    def grid_mapping(self):
        grid_mapping = self.var.attrs.get("grid_mapping", None)
        if grid_mapping is None:
            return None
        return self.ds[grid_mapping].attrs

    def grid_points(self):
        return self.grid.grid_points()

    @property
    def latitudes(self):
        return self.grid.latitudes

    @property
    def longitudes(self):
        return self.grid.longitudes

    def __repr__(self):
        return "Variable[name=%s,coordinates=%s,metadata=%s]" % (
            self.var.name,
            self.coordinates,
            self._metadata,
        )

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError(i)
        coords = np.unravel_index(i, self.shape)
        kwargs = {k: v for k, v in zip(self.names, coords)}
        return XArrayField(self, self.var.isel(kwargs))

    def _valid_datetime(self, step):
        return np.datetime64(self.forecast_reference_time + datetime.timedelta(hours=step))

    def sel(self, missing, **kwargs):

        if not kwargs:
            return self

        # print('sel', kwargs, self.by_name.keys())

        k, v = kwargs.popitem()

        if k == "valid_datetime":
            k = "time"

        c = self.by_name.get(k)

        if c is None:
            missing[k] = v
            return self.sel(missing, **kwargs)

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

        return variable.sel(missing, **kwargs)

    def match(self, **kwargs):
        for k, v in list(kwargs.items()):

            if not isinstance(v, list):
                v = [v]

            name = "variable" if k == "param" else k

            if name in self._metadata:
                if self._metadata[name] not in v:
                    return False, None

                kwargs.pop(k)

        # print("match", kwargs)
        return True, kwargs


class FilteredVariable:
    def __init__(self, variable, **kwargs):
        assert isinstance(variable, Variable)
        self.variable = variable
        self.kwargs = kwargs

    @cached_property
    def fields(self):
        result = []
        print(type(self.variable), self.variable, self.kwargs)
        for field in self.variable:
            if all(field.metadata(k) == v for k, v in self.kwargs.items()):
                result.append(field)
        return result
        # return [field for field in self.variable if all(field.metadata(k) == v for k, v in self.kwargs.items())]

    @property
    def length(self):
        return len(self.fields)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError(i)
        return self.fields[i]


class XarrayFieldList(FieldList):
    def __init__(self, ds, variables):
        self.ds = ds
        self.variables = variables.copy()
        self.total_length = sum(v.length for v in variables)

    def __repr__(self):
        return f"XarrayFieldList({self.total_length})"

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
                missing = {}
                v = v.sel(missing, **rest)
                if missing and v is not None:
                    v = FilteredVariable(v, **missing)

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
    return MultiFieldList([fs.sel(valid_datetime=date, **kwargs) for date in dates])
