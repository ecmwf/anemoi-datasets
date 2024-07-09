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

from .field import XArrayField

LOG = logging.getLogger(__name__)


class Variable:
    def __init__(self, *, ds, var, coordinates, grid, time, metadata, array_backend=None):
        self.ds = ds
        self.var = var

        self.grid = grid
        self.coordinates = coordinates

        self._metadata = metadata.copy()
        self._metadata.update(var.attrs)
        self._metadata.update({"variable": var.name})

        self._metadata.setdefault("level", None)
        self._metadata.setdefault("number", 0)
        self._metadata.setdefault("levtype", "sfc")

        self.time = time

        self.shape = tuple(len(c.variable) for c in coordinates if c.is_dim and not c.scalar and not c.is_grid)
        self.names = {c.variable.name: c for c in coordinates if c.is_dim and not c.scalar and not c.is_grid}
        self.by_name = {c.variable.name: c for c in coordinates}

        self.length = math.prod(self.shape)
        self.array_backend = ensure_backend(array_backend)

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
            time=self.time,
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
        self.variable = variable
        self.kwargs = kwargs

    @cached_property
    def fields(self):
        return [field for field in self.variable if all(field.metadata(k) == v for k, v in self.kwargs.items())]

    @property
    def length(self):
        return len(self.fields)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError(i)
        return self.fields[i]
