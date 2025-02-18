# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property
from typing import Any
from typing import Dict
from typing import Optional

from anemoi.utils.dates import as_datetime
from earthkit.data.core.geography import Geography
from earthkit.data.core.metadata import RawMetadata
from earthkit.data.utils.projections import Projection

LOG = logging.getLogger(__name__)


class _MDMapping:

    def __init__(self, variable: Any) -> None:
        self.variable = variable
        self.time = variable.time
        # Aliases
        self.mapping = dict(param="variable")
        for c in variable.coordinates:
            for v in c.mars_names:
                assert v not in self.mapping, f"Duplicate key '{v}' in {c}"
                self.mapping[v] = c.variable.name

    def _from_user(self, key: str) -> str:
        return self.mapping.get(key, key)

    def from_user(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {self._from_user(k): v for k, v in kwargs.items()}

    def __repr__(self) -> str:
        return f"MDMapping({self.mapping})"

    def fill_time_metadata(self, field: Any, md: Dict[str, Any]) -> None:
        valid_datetime = self.variable.time.fill_time_metadata(field._md, md)
        if valid_datetime is not None:
            md["valid_datetime"] = as_datetime(valid_datetime).isoformat()


class XArrayMetadata(RawMetadata):
    LS_KEYS = ["variable", "level", "valid_datetime", "units"]
    NAMESPACES = ["default", "mars"]
    MARS_KEYS = ["param", "step", "levelist", "levtype", "number", "date", "time"]

    def __init__(self, field: Any) -> None:
        self._field = field
        md = field._md.copy()
        self._mapping = _MDMapping(field.owner)
        self._mapping.fill_time_metadata(field, md)
        super().__init__(md)

    @cached_property
    def geography(self) -> "XArrayFieldGeography":
        return XArrayFieldGeography(self._field, self._field.owner.grid)

    def as_namespace(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(namespace, str) and namespace is not None:
            raise TypeError("namespace must be a str or None")

        if namespace == "default" or namespace == "" or namespace is None:
            return dict(self)

        elif namespace == "mars":
            return self._as_mars()

    def _as_mars(self) -> Dict[str, Any]:
        return {}

    def _base_datetime(self) -> Optional[datetime.datetime]:
        return self._field.forecast_reference_time

    def _valid_datetime(self) -> Optional[datetime.datetime]:
        return self._get("valid_datetime")

    def get(self, key: str, astype: Optional[type] = None, **kwargs: Any) -> Any:

        if key in self._d:
            if astype is not None:
                return astype(self._d[key])
            return self._d[key]

        key = self._mapping._from_user(key)

        return super().get(key, astype=astype, **kwargs)


class XArrayFieldGeography(Geography):
    def __init__(self, field: Any, grid: Any) -> None:
        self._field = field
        self._grid = grid

    def _unique_grid_id(self) -> None:
        raise NotImplementedError()

    def bounding_box(self) -> None:
        raise NotImplementedError()
        # return BoundingBox(north=self.north, south=self.south, east=self.east, west=self.west)

    def gridspec(self) -> None:
        raise NotImplementedError()

    def latitudes(self, dtype: Optional[type] = None) -> Any:
        result = self._grid.latitudes
        if dtype is not None:
            return result.astype(dtype)
        return result

    def longitudes(self, dtype: Optional[type] = None) -> Any:
        result = self._grid.longitudes
        if dtype is not None:
            return result.astype(dtype)
        return result

    def resolution(self) -> Optional[Any]:
        # TODO: implement resolution
        return None

    # @property
    def mars_grid(self) -> Optional[Any]:
        # TODO: implement mars_grid
        return None

    # @property
    def mars_area(self) -> Optional[Any]:
        # TODO: code me
        # return [self.north, self.west, self.south, self.east]
        return None

    def x(self, dtype: Optional[type] = None) -> None:
        raise NotImplementedError()

    def y(self, dtype: Optional[type] = None) -> None:
        raise NotImplementedError()

    def shape(self) -> Any:
        return self._field.shape

    def projection(self) -> Projection:
        return Projection.from_cf_grid_mapping(**self._field.grid_mapping)
