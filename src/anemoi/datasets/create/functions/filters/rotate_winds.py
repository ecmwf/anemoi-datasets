# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from collections import defaultdict

import tqdm
from anemoi.utils.humanize import plural
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.geo.rotate import rotate_vector


class NewDataField:
    def __init__(self, field, data):
        self.field = field
        self.data = data

    def to_numpy(self, *args, **kwargs):
        return self.data

    def __getattr__(self, name):
        return getattr(self.field, name)

    def __repr__(self) -> str:
        return repr(self.field)


def execute(
    context,
    input,
    x_wind,
    y_wind,
    source_projection=None,
    target_projection="+proj=longlat",
):
    from pyproj import CRS

    context.trace("🔄", "Rotating winds (extracting winds from ", plural(len(input), "field"))

    result = FieldArray()

    wind_params = (x_wind, y_wind)
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
