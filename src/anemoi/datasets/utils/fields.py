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
        self._field = field

    def __getattr__(self, name: str) -> Any:
        return getattr(self._field, name)

    def __repr__(self) -> str:
        return repr(self._field)


class NewDataField(WrappedField):
    """Class to represent a new data field with additional data."""

    def __init__(self, field: Any, data: Any) -> None:
        super().__init__(field)
        self._data = data
        self.shape = data.shape

    def to_numpy(self, flatten: bool = False, dtype: Optional[Any] = None, index: Optional[Any] = None) -> Any:
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
        super().__init__(field)
        self._metadata = kwargs

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and args[0] in self._metadata:
            return self._metadata[args[0]]
        return self._field.metadata(*args, **kwargs)
