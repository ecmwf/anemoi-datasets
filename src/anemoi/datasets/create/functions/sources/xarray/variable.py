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
    def __init__(
        self,
        *,
        ds,
        variable,
        coordinates,
        grid,
        time,
        metadata,
        array_backend=None,
    ):
        self.ds = ds
        self.variable = variable

        self.grid = grid
        self.coordinates = coordinates

        self._metadata = metadata.copy()
        self._metadata.update({"variable": variable.name})

        self.time = time

        self.shape = tuple(len(c.variable) for c in coordinates if c.is_dim and not c.scalar and not c.is_grid)
        self.names = {c.variable.name: c for c in coordinates if c.is_dim and not c.scalar and not c.is_grid}
        self.by_name = {c.variable.name: c for c in coordinates}

        self.length = math.prod(self.shape)
        self.array_backend = ensure_backend(array_backend)

    @property
    def name(self):
        return self.variable.name

    def __len__(self):
        return self.length

    @property
    def grid_mapping(self):
        grid_mapping = self.variable.attrs.get("grid_mapping", None)
        if grid_mapping is None:
            return None
        return self.ds[grid_mapping].attrs

    def grid_points(self):
        return self.grid.grid_points

    @property
    def latitudes(self):
        return self.grid.latitudes

    @property
    def longitudes(self):
        return self.grid.longitudes

    def __repr__(self):
        return "Variable[name=%s,coordinates=%s,metadata=%s]" % (
            self.variable.name,
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
        return XArrayField(self, self.variable.isel(kwargs))

    def sel(self, missing, **kwargs):

        if not kwargs:
            return self

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
            var=self.variable.isel({k: i}),
            coordinates=coordinates,
            grid=self.grid,
            time=self.time,
            metadata=metadata,
        )

        return variable.sel(missing, **kwargs)

    def match(self, **kwargs):

        if "param" in kwargs:
            assert "variable" not in kwargs
            kwargs["variable"] = kwargs.pop("param")

        if "variable" in kwargs:
            name = kwargs.pop("variable")
            if not isinstance(name, (list, tuple)):
                name = [name]
            if self.variable.name not in name:
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
