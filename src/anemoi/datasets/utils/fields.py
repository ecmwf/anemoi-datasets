# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Optional


class WrappedField:
    """Wrapper class for a field to provide additional functionality."""

    def __init__(self, field: Any) -> None:
        """Initialize the WrappedField.

        Parameters
        ----------
        field : Any
            The field to be wrapped.
        """
        self._field = field

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped field.

        Parameters
        ----------
        name : str
            The name of the attribute to get.

        Returns
        -------
        Any
            The attribute of the wrapped field.
        """
        return getattr(self._field, name)

    def __repr__(self) -> str:
        """Get the string representation of the wrapped field.

        Returns
        -------
        str
            The string representation of the wrapped field.
        """
        return repr(self._field)


class NewDataField(WrappedField):
    """Class to represent a new data field with additional data."""

    def __init__(self, field: Any, data: Any) -> None:
        """Initialize the NewDataField.

        Parameters
        ----------
        field : Any
            The field to be wrapped.
        data : Any
            The additional data for the field.
        """
        super().__init__(field)
        self._data = data
        self.shape = data.shape

    def to_numpy(self, flatten: bool = False, dtype: Optional[Any] = None, index: Optional[Any] = None) -> Any:
        """Convert the data to a numpy array.

        Parameters
        ----------
        flatten : bool, optional
            Whether to flatten the data, by default False.
        dtype : Optional[Any], optional
            The desired data type of the array, by default None.
        index : Optional[Any], optional
            The index to apply to the data, by default None.

        Returns
        -------
        Any
            The numpy array representation of the data.
        """
        data = self._data
        if dtype is not None:
            data = data.astype(dtype)
        if flatten:
            data = data.flatten()
        if index is not None:
            data = data[index]
        return data


class NewMetadataField(WrappedField):
    """Class to represent a new metadata field with additional metadata."""

    def __init__(self, field: Any, **kwargs: Any) -> None:
        """Initialize the NewMetadataField.

        Parameters
        ----------
        field : Any
            The field to be wrapped.
        **kwargs : Any
            Additional metadata for the field.
        """
        super().__init__(field)
        self._metadata = kwargs

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Retrieve metadata for the field.

        Parameters
        ----------
        *args : Any
            Positional arguments for metadata retrieval.
        **kwargs : Any
            Keyword arguments for metadata retrieval.

        Returns
        -------
        Any
            The metadata for the field.
        """
        if len(args) == 1 and args[0] in self._metadata:
            return self._metadata[args[0]]
        return self._field.metadata(*args, **kwargs)
