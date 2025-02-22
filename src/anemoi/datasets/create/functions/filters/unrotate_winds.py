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

import earthkit.data as ekd

# import numpy as np
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.geo.rotate import unrotate_vector


class NewDataField:
    """A class to represent a new data field with unrotated wind components."""

    def __init__(self, field: Any, data: Any) -> None:
        """
        Initialize a NewDataField instance.

        Parameters
        ----------
        field : Any
            The original field.
        data : Any
            The unrotated wind component data.
        """
        self.field = field
        self.data = data

    def to_numpy(self, *args: Any, **kwargs: Any) -> Any:
        """
        Convert the data to a numpy array.

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
        """
        Get an attribute from the original field.

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


def execute(context: Any, input: ekd.FieldList, u: str, v: str) -> FieldArray:
    """
    Unrotate the wind components of a GRIB file.

    Parameters
    ----------
    context : Any
        The execution context.
    input : List[Any]
        The list of input fields.
    u : str
        The parameter name for the u-component of the wind.
    v : str
        The parameter name for the v-component of the wind.

    Returns
    -------
    FieldArray
        The resulting field array with unrotated wind components.
    """
    result = FieldArray()

    wind_params = (u, v)
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

        x = pairs[u]
        y = pairs[v]

        lats, lons = x.grid_points()
        raw_lats, raw_longs = x.grid_points_unrotated()

        assert x.rotation == y.rotation

        u_new, v_new = unrotate_vector(
            lats,
            lons,
            x.to_numpy(flatten=True),
            y.to_numpy(flatten=True),
            *x.rotation[:2],
            south_pole_rotation_angle=x.rotation[2],
            lat_unrotated=raw_lats,
            lon_unrotated=raw_longs,
        )

        result.append(NewDataField(x, u_new))
        result.append(NewDataField(y, v_new))

    return result


if __name__ == "__main__":
    from earthkit.data import from_source

    source = from_source(
        "mars",
        date=-1,
        param="10u/10v",
        levtype="sfc",
        grid=[1, 1],
        area=[28, 0, -14, 40],
        rotation=[-22, -40],
    )

    source.save("source.grib")

    execute(None, source, "10u", "10v")
