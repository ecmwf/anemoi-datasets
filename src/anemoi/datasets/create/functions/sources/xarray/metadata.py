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

from earthkit.data.core.geography import Geography
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.utils.dates import to_datetime
from earthkit.data.utils.projections import Projection

LOG = logging.getLogger(__name__)


class MDMapping:

    def __init__(self, mapping):
        self.user_to_internal = mapping

    def from_user(self, kwargs):
        if isinstance(kwargs, str):
            return self.user_to_internal.get(kwargs, kwargs)
        return {self.user_to_internal.get(k, k): v for k, v in kwargs.items()}

    def __len__(self):
        return len(self.user_to_internal)

    def __repr__(self):
        return f"MDMapping({self.user_to_internal})"


class XArrayMetadata(RawMetadata):
    LS_KEYS = ["variable", "level", "valid_datetime", "units"]
    NAMESPACES = ["default", "mars"]
    MARS_KEYS = ["param", "step", "levelist", "levtype", "number", "date", "time"]

    def __init__(self, field, mapping):
        self._field = field
        md = field._md.copy()

        self._mapping = mapping
        if mapping is None:
            time_coord = [c for c in field.owner.coordinates if c.is_time]
            if len(time_coord) == 1:
                time_key = time_coord[0].name
            else:
                time_key = "time"
        else:
            time_key = mapping.from_user("valid_datetime")
        self._time = to_datetime(md.pop(time_key))
        self._field.owner.time.fill_time_metadata(self._time, md)
        md["valid_datetime"] = self._time.isoformat()

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

        if self._mapping is not None:
            key = self._mapping.from_user(key)

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
