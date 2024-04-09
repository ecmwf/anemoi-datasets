# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from collections import defaultdict

from climetlab.indexing.fieldset import FieldArray


def rotate_winds(lats, lons, x_wind, y_wind, source_projection, target_projection):
    """
    Code provided by MetNO
    """
    import numpy as np
    import pyproj

    if source_projection == target_projection:
        return x_wind, x_wind

    source_projection = pyproj.Proj(source_projection)
    target_projection = pyproj.Proj(target_projection)

    transformer = pyproj.transformer.Transformer.from_proj(source_projection, target_projection)

    # To compute the new vector components:
    # 1) perturb each position in the direction of the winds
    # 2) convert the perturbed positions into the new coordinate system
    # 3) measure the new x/y components.
    #
    # A complication occurs when using the longlat "projections", since this is not a cartesian grid
    # (i.e. distances in each direction is not consistent), we need to deal with the fact that the
    # width of a longitude varies with latitude
    orig_speed = np.sqrt(x_wind**2 + y_wind**2)

    x0, y0 = source_projection(lons, lats)

    if source_projection.name != "longlat":
        x1 = x0 + x_wind
        y1 = y0 + y_wind
    else:
        # Reduce the perturbation, since x_wind and y_wind are in meters, which would create
        # large perturbations in lat, lon. Also, deal with the fact that the width of longitude
        # varies with latitude.
        factor = 3600000.0
        x1 = x0 + x_wind / factor / np.cos(np.deg2rad(lats))
        y1 = y0 + y_wind / factor

    X0, Y0 = transformer.transform(x0, y0)
    X1, Y1 = transformer.transform(x1, y1)

    new_x_wind = X1 - X0
    new_y_wind = Y1 - Y0
    if target_projection.name == "longlat":
        new_x_wind *= np.cos(np.deg2rad(lats))

    if target_projection.name == "longlat" or source_projection.name == "longlat":
        # Ensure the wind speed is not changed (which might not the case since the units in longlat
        # is degrees, not meters)
        curr_speed = np.sqrt(new_x_wind**2 + new_y_wind**2)
        new_x_wind *= orig_speed / curr_speed
        new_y_wind *= orig_speed / curr_speed

    return new_x_wind, new_y_wind


class NewDataField:
    def __init__(self, field, data):
        self.field = field
        self.data = data

    def to_numpy(self, *args, **kwargs):
        return self.data

    def __getattr__(self, name):
        return getattr(self.field, name)


def execute(
    context,
    input,
    x_wind,
    y_wind,
    source_projection=None,
    target_projection="+proj=longlat",
):
    from pyproj import CRS

    result = FieldArray()

    wind_params = (x_wind, y_wind)
    wind_pairs = defaultdict(dict)

    for f in input:
        key = f.as_mars()
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

        x = pairs[x_wind]
        y = pairs[y_wind]

        assert x.grid_mapping == y.grid_mapping

        lats, lons = x.grid_points()
        x_new, y_new = rotate_winds(
            lats,
            lons,
            x.to_numpy(reshape=False),
            y.to_numpy(reshape=False),
            (source_projection if source_projection is not None else CRS.from_cf(x.grid_mapping)),
            target_projection,
        )

        result.append(NewDataField(x, x_new))
        result.append(NewDataField(y, y_new))

    return result
