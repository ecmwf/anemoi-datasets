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

import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo.wind.array import polar_to_xy


class NewDataField:
    """
    A class to represent a new data field with converted wind component data.

    Attributes:
        field (Any): The original field.
        data (Any): The converted wind component data.
        new_name (str): The new name for the field.
    """

    def __init__(self, field: Any, data: Any, new_name: str) -> None:
        """
        Initialize a NewDataField instance.

        Args:
            field (Any): The original field.
            data (Any): The converted wind component data.
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
        Get metadata from the original field, with the option to rename the parameter.

        Args:
            key (Optional[str]): The metadata key.
            **kwargs (Any): Additional keyword arguments.

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
        Get an attribute from the original field.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute value.
        """
        return getattr(self.field, name)


def execute(
    context: Any,
    input: List[Any],
    wind_speed: str,
    wind_dir: str,
    u_component: str = "u",
    v_component: str = "v",
    in_radians: bool = False,
) -> FieldArray:

    result = FieldArray()

    wind_params = (wind_speed, wind_dir)
    wind_pairs = defaultdict(dict)

    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")

        if param not in wind_params:
            result.append(f)
            continue

        key = tuple(key.items())

        if param in wind_pairs[key]:
            raise ValueError(f"Duplicate wind component {param} for {key}")

        wind_pairs[key][param] = f

    for _, pairs in wind_pairs.items():
        if len(pairs) != 2:
            raise ValueError("Missing wind component")

        magnitude = pairs[wind_speed]
        direction = pairs[wind_dir]

        # assert speed.grid_mapping == dir.grid_mapping
        if in_radians:
            direction = np.rad2deg(direction)

        u, v = polar_to_xy(magnitude.to_numpy(flatten=True), direction.to_numpy(flatten=True))

        result.append(NewDataField(magnitude, u, u_component))
        result.append(NewDataField(direction, v, v_component))

    return result
