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
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple

import earthkit.data as ekd
from earthkit.data.indexing.fieldlist import FieldArray


class NewDataField:
    """Represents a new data field with modified data and metadata.

    Attributes
    ----------
    field : Any
        The original field.
    data : Any
        The data for the new field.
    new_name : str
        The new name for the field.
    """

    def __init__(self, field: Any, data: Any, new_name: str) -> None:
        """
        Initialize a NewDataField instance.

        Parameters
        ----------
        field : Any
            The original field.
        data : Any
            The data for the new field.
        new_name : str
            The new name for the field.
        """
        self.field = field
        self.data = data
        self.new_name = new_name

    def to_numpy(self, *args: Any, **kwargs: Any) -> Any:
        """
        Convert the data to a numpy array.

        Parameters
        ----------
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The data as a numpy array.
        """
        return self.data

    def metadata(self, key: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Retrieve metadata for the field.

        Parameters
        ----------
        key : Optional[str], optional
            The metadata key to retrieve. If None, all metadata is returned.
        **kwargs : Any
            Additional arguments for metadata retrieval.

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
        """
        Delegate attribute access to the original field.

        Parameters
        ----------
        name : str
            The attribute name.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self.field, name)


def execute(context: Any, input: ekd.FieldList, params: List[str], output: str) -> ekd.FieldList:
    """Computes the sum over a set of variables.

    Args:
        context (Any): The execution context.
        input (List[Any]): The list of input fields.
        params (List[str]): The list of parameters to sum over.
        output (str): The name for the output field.

    Returns:
        ekd.FieldList: The resulting FieldArray with summed fields.
    """
    result = FieldArray()

    needed_fields: Dict[Tuple[Hashable, ...], Dict[str, ekd.Field]] = defaultdict(dict)

    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        if param in params:
            key = tuple(key.items())

            if param in needed_fields[key]:
                raise ValueError(f"Duplicate field {param} for {key}")

            needed_fields[key][param] = f
        else:
            result.append(f)

    for keys, values in needed_fields.items():

        if len(values) != len(params):
            raise ValueError("Missing fields")

        s = None
        for k, v in values.items():
            c = v.to_numpy(flatten=True)
            if s is None:
                s = c
            else:
                s += c
        result.append(NewDataField(values[list(values.keys())[0]], s, output))

    return result
