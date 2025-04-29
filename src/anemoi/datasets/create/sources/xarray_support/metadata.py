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
    """A class to handle metadata mapping for variables.

    Attributes
    ----------
    variable : Any
        The variable to map.
    time : Any
        The time associated with the variable.
    mapping : Dict[str, str]
        A dictionary mapping keys to variable names.
    """

    def __init__(self, variable: Any) -> None:
        """Initialize the _MDMapping class.

        Parameters
        ----------
        variable : Any
            The variable to map.
        """
        self.variable = variable
        self.time = variable.time
        self.mapping = dict()
        # Aliases

    def _from_user(self, key: str) -> str:
        """Get the internal key corresponding to a user-provided key.

        Parameters
        ----------
        key : str
            The user-provided key.

        Returns
        -------
        str
            The internal key corresponding to the user-provided key.
        """
        return self.mapping.get(key, key)

    def from_user(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert user-provided keys to internal keys.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            A dictionary of user-provided keys and values.

        Returns
        -------
        Dict[str, Any]
            A dictionary with internal keys and original values.
        """
        return {self._from_user(k): v for k, v in kwargs.items()}

    def __repr__(self) -> str:
        """Return a string representation of the _MDMapping object.

        Returns
        -------
        str
            String representation of the _MDMapping object.
        """
        return f"MDMapping({self.mapping})"

    def fill_time_metadata(self, field: Any, md: Dict[str, Any]) -> None:
        """Fill the time metadata for a field.

        Parameters
        ----------
        field : Any
            The field to fill metadata for.
        md : Dict[str, Any]
            The metadata dictionary to update.
        """
        valid_datetime = self.variable.time.fill_time_metadata(field._md, md)
        if valid_datetime is not None:
            md["valid_datetime"] = as_datetime(valid_datetime).isoformat()


class XArrayMetadata(RawMetadata):
    """A class to handle metadata for XArray fields.

    Attributes
    ----------
    LS_KEYS : List[str]
        List of keys for the metadata.
    NAMESPACES : List[str]
        List of namespaces for the metadata.
    MARS_KEYS : List[str]
        List of MARS keys for the metadata.
    """

    LS_KEYS = ["variable", "level", "valid_datetime", "units"]
    NAMESPACES = ["default", "mars"]
    MARS_KEYS = ["param", "step", "levelist", "levtype", "number", "date", "time"]

    def __init__(self, field: Any) -> None:
        """Initialize the XArrayMetadata class.

        Parameters
        ----------
        field : Any
            The field to extract metadata from.
        """
        self._field = field
        md = field._md.copy()
        self._mapping = _MDMapping(field.owner)
        self._mapping.fill_time_metadata(field, md)
        super().__init__(md)

    @cached_property
    def geography(self) -> "XArrayFieldGeography":
        """Get the geography information for the field."""
        return XArrayFieldGeography(self._field, self._field.owner.grid)

    def as_namespace(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get the metadata as a specific namespace.

        Parameters
        ----------
        namespace : Optional[str]
            The namespace to use.

        Returns
        -------
        Dict[str, Any]
            The metadata in the specified namespace.
        """
        if not isinstance(namespace, str) and namespace is not None:
            raise TypeError("namespace must be a str or None")

        if namespace == "default" or namespace == "" or namespace is None:
            return dict(self)

        elif namespace == "mars":
            return self._as_mars()

    def _as_mars(self) -> Dict[str, Any]:
        """Get the metadata as MARS namespace.

        Returns
        -------
        Dict[str, Any]
            The metadata in the MARS namespace.
        """
        return {}

    def _base_datetime(self) -> Optional[datetime.datetime]:
        """Get the base datetime for the field.

        Returns
        -------
        Optional[datetime.datetime]
            The base datetime for the field.
        """
        return self._field.forecast_reference_time

    def _valid_datetime(self) -> Optional[datetime.datetime]:
        """Get the valid datetime for the field.

        Returns
        -------
        Optional[datetime.datetime]
            The valid datetime for the field.
        """
        return self._get("valid_datetime")

    def get(self, key: str, astype: Optional[type] = None, **kwargs: Any) -> Any:
        """Get a metadata value by key.

        Parameters
        ----------
        key : str
            The key to get the value for.
        astype : Optional[type]
            The type to cast the value to.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The value for the specified key, optionally cast to the specified type.
        """

        if key == "levelist":
            # Special case for levelist, for compatibility with GRIB
            if key not in self._d and "level" in self._d:
                key = "level"

        if key in self._d:
            if astype is not None:
                return astype(self._d[key])
            return self._d[key]

        key = self._mapping._from_user(key)

        return super().get(key, astype=astype, **kwargs)


class XArrayFieldGeography(Geography):
    """A class to handle geography information for XArray fields.

    Attributes
    ----------
    _field : Any
        The field to extract geography information from.
    _grid : Any
        The grid associated with the field.
    """

    def __init__(self, field: Any, grid: Any) -> None:
        """Initialize the XArrayFieldGeography class.

        Parameters
        ----------
        field : Any
            The field to extract geography information from.
        grid : Any
            The grid associated with the field.
        """
        self._field = field
        self._grid = grid

    def _unique_grid_id(self) -> None:
        """Get the unique grid ID.

        Raises
        ------
        NotImplementedError
            This method is not implemented.
        """
        raise NotImplementedError()

    def bounding_box(self) -> None:
        """Get the bounding box for the field.

        Raises
        ------
        NotImplementedError
            This method is not implemented.
        """
        raise NotImplementedError()
        # return BoundingBox(north=self.north, south=self.south, east=self.east, west=self.west)

    def gridspec(self) -> None:
        """Get the grid specification for the field.

        Raises
        ------
        NotImplementedError
            This method is not implemented.
        """
        raise NotImplementedError()

    def latitudes(self, dtype: Optional[type] = None) -> Any:
        """Get the latitudes for the field.

        Parameters
        ----------
        dtype : Optional[type]
            The type to cast the latitudes to.

        Returns
        -------
        Any
            The latitudes for the field.
        """
        result = self._grid.latitudes
        if dtype is not None:
            return result.astype(dtype)
        return result

    def longitudes(self, dtype: Optional[type] = None) -> Any:
        """Get the longitudes for the field.

        Parameters
        ----------
        dtype : Optional[type]
            The type to cast the longitudes to.

        Returns
        -------
        Any
            The longitudes for the field.
        """
        result = self._grid.longitudes
        if dtype is not None:
            return result.astype(dtype)
        return result

    def resolution(self) -> Optional[Any]:
        """Get the resolution for the field.

        Returns
        -------
        Optional[Any]
            The resolution for the field.
        """
        # TODO: implement resolution
        return None

    def mars_grid(self) -> Optional[Any]:
        """Get the MARS grid for the field.

        Returns
        -------
        Optional[Any]
            The MARS grid for the field.
        """
        # TODO: implement mars_grid
        return None

    def mars_area(self) -> Optional[Any]:
        """Get the MARS area for the field.

        Returns
        -------
        Optional[Any]
            The MARS area for the field.
        """
        # TODO: code me
        # return [self.north, self.west, self.south, self.east]
        return None

    def x(self, dtype: Optional[type] = None) -> None:
        """Get the x-coordinates for the field.

        Parameters
        ----------
        dtype : Optional[type]
            The type to cast the x-coordinates to.

        Raises
        ------
        NotImplementedError
            This method is not implemented.
        """
        raise NotImplementedError()

    def y(self, dtype: Optional[type] = None) -> None:
        """Get the y-coordinates for the field.

        Parameters
        ----------
        dtype : Optional[type]
            The type to cast the y-coordinates to.

        Raises
        ------
        NotImplementedError
            This method is not implemented.
        """
        raise NotImplementedError()

    def shape(self) -> Any:
        """Get the shape of the field.

        Returns
        -------
        Any
            The shape of the field.
        """
        return self._field.shape

    def projection(self) -> Projection:
        """Get the projection for the field.

        Returns
        -------
        Projection
            The projection for the field.
        """
        return Projection.from_cf_grid_mapping(**self._field.grid_mapping)
