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
from functools import cached_property

import numpy as np
from earthkit.data.utils.array import ensure_backend

from anemoi.datasets.create.functions.sources.xarray.metadata import MDMapping

from .field import XArrayField

LOG = logging.getLogger(__name__)


class Variable:
    def __init__(self, *, ds, var, coordinates, grid, time, metadata, mapping=None, array_backend=None):
        self.ds = ds
        self.var = var

        self.grid = grid
        self.coordinates = coordinates

        # print("Variable", var.name)
        # for c in coordinates:
        #     print(" ", c)

        self._metadata = metadata.copy()
        # self._metadata.update(var.attrs)
        self._metadata.update({"variable": var.name})

        # self._metadata.setdefault("level", None)
        # self._metadata.setdefault("number", 0)
        # self._metadata.setdefault("levtype", "sfc")
        self._mapping = mapping

        self.time = time

        self.shape = tuple(len(c.variable) for c in coordinates if c.is_dim and not c.scalar and not c.is_grid)
        self.names = {c.variable.name: c for c in coordinates if c.is_dim and not c.scalar and not c.is_grid}
        self.by_name = {c.variable.name: c for c in coordinates}

        self.length = math.prod(self.shape)
        self.array_backend = ensure_backend(array_backend)

    def update_metadata_mapping(self, kwargs):

        result = {}

        for k, v in kwargs.items():
            if k == "param":
                result[k] = "variable"
                continue

            for c in self.coordinates:
                if k in c.mars_names:
                    for v in c.mars_names:
                        result[v] = c.variable.name
                    break

        self._mapping = MDMapping(result)

    @property
    def name(self):
        return self.var.name

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
        """
        Get a 2D field from the variable
        """
        if i >= self.length:
            raise IndexError(i)

        coords = np.unravel_index(i, self.shape)
        kwargs = {k: v for k, v in zip(self.names, coords)}
        return XArrayField(self, self.var.isel(kwargs))

    @property
    def mapping(self):
        return self._mapping

    def sel(self, missing, **kwargs):

        if not kwargs:
            return self

        kwargs = self._mapping.from_user(kwargs)

        k, v = kwargs.popitem()

        c = self.by_name.get(k)

        if c is None:
            missing[k] = v
            return self.sel(missing, **kwargs)

        i = c.index(v)
        if i is None:
            LOG.warning(f"Could not find {k}={v} in {c}")
            return None

        coordinates = [x.reduced(i) if c is x else x for x in self.coordinates]

        metadata = self._metadata.copy()
        metadata.update({k: v})

        variable = Variable(
            ds=self.ds,
            var=self.var.isel({k: i}),
            coordinates=coordinates,
            grid=self.grid,
            time=self.time,
            metadata=metadata,
            mapping=self.mapping,
        )

        return variable.sel(missing, **kwargs)

    def match(self, **kwargs):
        kwargs = self._mapping.from_user(kwargs)

        if "variable" in kwargs:
            name = kwargs.pop("variable")
            if not isinstance(name, (list, tuple)):
                name = [name]
            if self.var.name not in name:
                return False, None
            return True, kwargs
        return True, kwargs


class FilteredVariable:
    def __init__(self, variable, **kwargs):
        self.variable = variable
        self.kwargs = kwargs

    @cached_property
    def fields(self):
        """Filter the fields of a variable based on metadata.

        Returns
        -------
        list
            A list of fields that match the metadata.
        """
        return [
            field
            for field in self.variable
            if all(field.metadata(k, default=None) == v for k, v in self.kwargs.items())
        ]

    @property
    def length(self):
        return len(self.fields)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError(i)
        return self.fields[i]
