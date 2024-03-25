# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from collections import defaultdict

import numpy as np
from climetlab.indexing.fieldset import FieldArray


def normalise(x):
    return max(min(x, 1.0), -1.0)


def normalise_longitude(lon, minimum):
    while lon < minimum:
        lon += 360

    while lon >= minimum + 360:
        lon -= 360

    return lon


def rotate_winds(
    lats,
    lons,
    raw_lats,
    raw_lons,
    x_wind,
    y_wind,
    south_pole_latitude,
    south_pole_longitude,
    south_pole_rotation_angle=0,
):
    # Code from MIR
    assert south_pole_rotation_angle == 0
    C = np.deg2rad(90 - south_pole_latitude)
    cos_C = np.cos(C)
    sin_C = np.sin(C)

    new_x = np.zeros_like(x_wind)
    new_y = np.zeros_like(y_wind)

    for i, (vx, vy, lat, lon, raw_lat, raw_lon) in enumerate(zip(x_wind, y_wind, lats, lons, raw_lats, raw_lons)):
        lonRotated = south_pole_longitude - lon
        lon_rotated = normalise_longitude(lonRotated, -180)
        lon_unrotated = raw_lon

        a = np.deg2rad(lon_rotated)
        b = np.deg2rad(lon_unrotated)
        q = 1 if (sin_C * lon_rotated < 0.0) else -1.0  # correct quadrant

        cos_c = normalise(np.cos(a) * np.cos(b) + np.sin(a) * np.sin(b) * cos_C)
        sin_c = q * np.sqrt(1.0 - cos_c * cos_c)

        new_x[i] = cos_c * vx + sin_c * vy
        new_y[i] = -sin_c * vx + cos_c * vy

    return new_x, new_y


class NewDataField:
    def __init__(self, field, data):
        self.field = field
        self.data = data

    def to_numpy(self, *args, **kwargs):
        return self.data

    def __getattr__(self, name):
        return getattr(self.field, name)


def execute(context, input, u, v):
    """
    Unrotate the wind components of a GRIB file.
    """
    result = FieldArray()

    wind_params = (u, v)
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

        x = pairs[u]
        y = pairs[v]

        lats, lons = x.grid_points()
        raw_lats, raw_longs = x.grid_points_raw()

        assert x.rotation == y.rotation

        u_new, v_new = rotate_winds(
            lats,
            lons,
            raw_lats,
            raw_longs,
            x.to_numpy(reshape=False),
            y.to_numpy(reshape=False),
            *x.rotation,
        )

        result.append(NewDataField(x, u_new))
        result.append(NewDataField(y, v_new))

    return result


if __name__ == "__main__":
    from climetlab import load_source

    source = load_source(
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
