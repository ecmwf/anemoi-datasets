# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict
from typing import Any
from typing import Dict

import earthkit.data as ekd
from anemoi.transform.field import new_field_from_numpy
from earthkit.data.indexing.fieldlist import FieldArray


class NewDataField:
    """A class to represent a new data field with modified data and metadata.

    Attributes
    ----------
    field : Any
        The original field.
    data : Any
        The data for the new field.
    new_name : str
        The new name for the parameter.
    """

    def __init__(self, field: Any, data: Any, new_name: str):
        """Initialize a new data field.

        Parameters
        ----------
        field : Any
            The original field.
        data : Any
            The data for the new field.
        new_name : str
            The new name for the parameter.
        """
        self.field: Any = field
        self.data: Any = data
        self.new_name: str = new_name

    def to_numpy(self, *args: Any, **kwargs: Any) -> Any:
        """Convert the data to a numpy array.

        Parameters
        ----------
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The data as a numpy array.
        """
        return self.data

    def metadata(self, key: str = None, **kwargs: Any) -> Any:
        """Retrieve metadata for the field.

        Parameters
        ----------
        key : str, optional
            The metadata key to retrieve. Defaults to None.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The metadata value.
        """
        if key is None:
            return self.field.metadata(**kwargs)
        value = self.field.metadata(key, **kwargs)
        if key == "param":
            return self.new_name
        return value

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the original field.

        Parameters
        ----------
        name : str
            The name of the attribute.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self.field, name)


def execute(context: Any, input: ekd.FieldList, orog: str, z: str = "z") -> ekd.FieldList:
    """Convert orography [m] to z (geopotential height).

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : FieldList
        List of input fields.
    orog : str
        Orography parameter.
    z : str, optional
        Geopotential height parameter. Defaults to "z".

    Returns
    -------
    FieldList
        List of fields with geopotential height.
    """
    result = FieldArray()
    processed_fields: Dict[tuple, Dict[str, Any]] = defaultdict(dict)

    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        if param == orog:
            key = tuple(key.items())

            if param in processed_fields[key]:
                raise ValueError(f"Duplicate field {param} for {key}")

            output = f.to_numpy(flatten=True) * 9.80665
            result.append(new_field_from_numpy(f, output, param=z))
        else:
            result.append(f)

    return result
