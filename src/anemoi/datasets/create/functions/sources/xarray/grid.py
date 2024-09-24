# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


from functools import cached_property

import numpy as np


class Grid:

    @property
    def latitudes(self):
        return self.grid_points[0]

    @property
    def longitudes(self):
        return self.grid_points[1]


class LatLonGrid(Grid):
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon


class XYGrid(Grid):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MeshedGrid(LatLonGrid):

    @cached_property
    def grid_points(self):
        return np.meshgrid(
            self.lat.variable.values,
            self.lon.variable.values,
        )


class UnstructuredGrid(LatLonGrid):

    @cached_property
    def grid_points(self):
        lat = self.lat.variable.values.flatten()
        lon = self.lon.variable.values.flatten()
        return lat, lon


class ProjectionGrid(XYGrid):
    def __init__(self, x, y, projection):
        super().__init__(x, y)
        self.projection = projection

    @cached_property
    def grid_points(self):
        from pyproj import CRS
        from pyproj import Transformer

        data_crs = CRS.from_cf(self.projection)
        wgs84_crs = CRS.from_epsg(4326)  # WGS84
        transformer = Transformer.from_crs(data_crs, wgs84_crs)
        lat, lon = transformer.transform(
            self.x.variable.values.flatten(),
            self.y.variable.values.flatten(),
        )

        assert False, (len(lat), len(lon))
        return np.meshgrid(lat, lon)
