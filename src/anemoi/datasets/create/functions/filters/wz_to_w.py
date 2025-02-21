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
from typing import List
from typing import Optional

from earthkit.data.indexing.fieldlist import FieldArray


class NewDataField:
    """
    A class to represent a new data field with modified data and metadata.

    Attributes:
        field (Any): The original field.
        data (Any): The data for the new field.
        new_name (str): The new name for the field.
    """

    def __init__(self, field: Any, data: Any, new_name: str) -> None:
        """
        Initialize a NewDataField instance.

        Args:
            field (Any): The original field.
            data (Any): The data for the new field.
            new_name (str): The new name for the field.
        """
        self.field = field
        self.data = data
        self.new_name = new_name

    def to_numpy(self, *args: Any, **kwargs: Any) -> Any:
        """
        Convert the data to a numpy array.

        Returns:
            Any: The data as a numpy array.
        """
        return self.data

    def metadata(self, key: Optional[str] = None, **kwargs: Any) -> Any:
        """
        Retrieve metadata for the field.

        Args:
            key (Optional[str]): The metadata key to retrieve. If None, all metadata is returned.
            **kwargs (Any): Additional arguments for metadata retrieval.

        Returns:
            Any: The metadata value.
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

        Args:
            name (str): The attribute name.

        Returns:
            Any: The attribute value.
        """
        return getattr(self.field, name)


def execute(context: Any, input: List[Any], wz: str, t: str, w: str = "w") -> FieldArray:
    """
    Convert geometric vertical velocity (m/s) to vertical velocity (Pa / s).

    Args:
        context (Any): The context for the execution.
        input (List[Any]): The list of input fields.
        wz (str): The parameter name for geometric vertical velocity.
        t (str): The parameter name for temperature.
        w (str, optional): The parameter name for vertical velocity. Defaults to "w".

    Returns:
        FieldArray: The resulting FieldArray with converted vertical velocity fields.
    """
    result = FieldArray()

    params = (wz, t)
    pairs = defaultdict(dict)

    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        if param in params:
            key = tuple(key.items())

            if param in pairs[key]:
                raise ValueError(f"Duplicate field {param} for {key}")

            pairs[key][param] = f
            if param == t:
                result.append(f)
        else:
            result.append(f)

    for keys, values in pairs.items():

        if len(values) != 2:
            raise ValueError("Missing fields")

        wz_pl = values[wz].to_numpy(flatten=True)
        t_pl = values[t].to_numpy(flatten=True)
        pressure = keys[4][1] * 100  # TODO: REMOVE HARDCODED INDICES

        w_pl = wz_to_w(wz_pl, t_pl, pressure)
        result.append(NewDataField(values[wz], w_pl, w))

    return result


def wz_to_w(wz: Any, t: Any, pressure: float) -> Any:
    """
    Convert geometric vertical velocity (m/s) to vertical velocity (Pa / s).

    Args:
        wz (Any): The geometric vertical velocity data.
        t (Any): The temperature data.
        pressure (float): The pressure value.

    Returns:
        Any: The vertical velocity data in Pa / s.
    """
    g = 9.81
    Rd = 287.058

    return -wz * g * pressure / (t * Rd)
