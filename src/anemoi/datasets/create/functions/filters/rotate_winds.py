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
from typing import List
from typing import Optional

import tqdm
from anemoi.utils.humanize import plural
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.geo.rotate import rotate_vector


class NewDataField:
    """A class to represent a new data field with rotated wind components."""

    def __init__(self, field: Any, data: Any) -> None:
        """Initialize a NewDataField instance.

        Parameters
        ----------
        field : Any
            The original field.
        data : Any
            The rotated wind component data.
        """
        self.field = field
        self.data = data

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

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the original field.

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

    def __repr__(self) -> str:
        """Get the string representation of the original field.

        Returns
        -------
        str
            The string representation of the original field.
        """
        return repr(self.field)


def execute(
    context: Any,
    input: List[Any],
    x_wind: str,
    y_wind: str,
    source_projection: Optional[str] = None,
    target_projection: str = "+proj=longlat",
) -> FieldArray:
    """Rotate wind components from one projection to another.

    Parameters
    ----------
    context : Any
        The context in which the function is executed.
    input : List[Any]
        List of input fields.
    x_wind : str
        X wind component parameter.
    y_wind : str
        Y wind component parameter.
    source_projection : Optional[str], optional
        Source projection, by default None.
    target_projection : str, optional
        Target projection, by default "+proj=longlat".

    Returns
    -------
    FieldArray
        Array of fields with rotated wind components.
    """
    from pyproj import CRS

    context.trace("🔄", "Rotating winds (extracting winds from ", plural(len(input), "field"))

    result = FieldArray()

    wind_params: tuple[str, str] = (x_wind, y_wind)
    wind_pairs: Dict[tuple, Dict[str, Any]] = defaultdict(dict)

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

    context.trace("🔄", "Rotating", plural(len(wind_pairs), "wind"), "(speed will likely include data download)")

    for _, pairs in tqdm.tqdm(list(wind_pairs.items())):
        if len(pairs) != 2:
            raise ValueError("Missing wind component")

        x = pairs[x_wind]
        y = pairs[y_wind]

        assert x.grid_mapping == y.grid_mapping

        lats, lons = x.grid_points()
        x_new, y_new = rotate_vector(
            lats,
            lons,
            x.to_numpy(flatten=True),
            y.to_numpy(flatten=True),
            (source_projection if source_projection is not None else CRS.from_cf(x.grid_mapping)),
            target_projection,
        )

        result.append(NewDataField(x, x_new))
        result.append(NewDataField(y, y_new))

    return result
