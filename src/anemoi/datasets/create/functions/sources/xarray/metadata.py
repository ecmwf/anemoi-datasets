# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from functools import cached_property

from anemoi.utils.dates import as_datetime
from earthkit.data.core.geography import Geography
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.utils.projections import Projection

LOG = logging.getLogger(__name__)


class _MDMapping:

    def __init__(self, variable):
        self.variable = variable
        self.time = variable.time
        self.mapping = dict(param="variable")
        for c in variable.coordinates:
            for v in c.mars_names:
                assert v not in self.mapping, f"Duplicate key '{v}' in {c}"
                self.mapping[v] = c.variable.name

    def _from_user(self, key):
        return self.mapping.get(key, key)

    def from_user(self, kwargs):
        print("from_user", kwargs, self)
        return {self._from_user(k): v for k, v in kwargs.items()}

    def __repr__(self):
        return f"MDMapping({self.mapping})"

    def fill_time_metadata(self, field, md):
        valid_datetime = self.variable.time.fill_time_metadata(field._md, md)
        if valid_datetime is not None:
            md["valid_datetime"] = as_datetime(valid_datetime).isoformat()


class XArrayMetadata(RawMetadata):
    LS_KEYS = ["variable", "level", "valid_datetime", "units"]
    NAMESPACES = ["default", "mars"]
    MARS_KEYS = ["param", "step", "levelist", "levtype", "number", "date", "time"]

    def __init__(self, field):
        self._field = field
        md = field._md.copy()
        self._mapping = _MDMapping(field.owner)
        self._mapping.fill_time_metadata(field, md)
        super().__init__(md)

    @cached_property
    def geography(self):
        return XArrayFieldGeography(self._field, self._field.owner.grid)

    def as_namespace(self, namespace=None):
        if not isinstance(namespace, str) and namespace is not None:
            raise TypeError("namespace must be a str or None")

        if namespace == "default" or namespace == "" or namespace is None:
            return dict(self)

        elif namespace == "mars":
            return self._as_mars()

    def _as_mars(self):
        return {}

    def _base_datetime(self):
        return self._field.forecast_reference_time

    def _valid_datetime(self):
        return self._get("valid_datetime")

    def _get(self, key, **kwargs):

        if key in self._d:
            return self._d[key]

        if key.startswith("mars."):
            key = key[5:]
            if key not in self.MARS_KEYS:
                if kwargs.get("raise_on_missing", False):
                    raise KeyError(f"Invalid key '{key}' in namespace='mars'")
                else:
                    return kwargs.get("default", None)

        key = self._mapping._from_user(key)

        return super()._get(key, **kwargs)


class XArrayFieldGeography(Geography):
    def __init__(self, field, grid):
        self._field = field
        self._grid = grid

    def _unique_grid_id(self):
        raise NotImplementedError()

    def bounding_box(self):
        raise NotImplementedError()
        # return BoundingBox(north=self.north, south=self.south, east=self.east, west=self.west)

    def gridspec(self):
        raise NotImplementedError()

    def latitudes(self, dtype=None):
        result = self._grid.latitudes
        if dtype is not None:
            return result.astype(dtype)
        return result

    def longitudes(self, dtype=None):
        result = self._grid.longitudes
        if dtype is not None:
            return result.astype(dtype)
        return result

    def resolution(self):
        # TODO: implement resolution
        return None

    # @property
    def mars_grid(self):
        # TODO: implement mars_grid
        return None

    # @property
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
