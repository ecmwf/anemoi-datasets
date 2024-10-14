# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
from functools import cached_property

import numpy as np

LOG = logging.getLogger(__name__)


class Grid:

    def __init__(self):
        pass

    @property
    def latitudes(self):
        return self.grid_points[0]

    @property
    def longitudes(self):
        return self.grid_points[1]


class LatLonGrid(Grid):
    def __init__(self, lat, lon, variable_dims):
        super().__init__()
        self.lat = lat
        self.lon = lon


class XYGrid(Grid):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MeshedGrid(LatLonGrid):

    @cached_property
    def grid_points(self):

        lat, lon = np.meshgrid(
            self.lat.variable.values,
            self.lon.variable.values,
        )

        return lat.flatten(), lon.flatten()


class UnstructuredGrid(LatLonGrid):

    def __init__(self, lat, lon, variable_dims):
        super().__init__(lat, lon, variable_dims)
        assert len(lat) == len(lon), (len(lat), len(lon))
        self.variable_dims = variable_dims
        self.grid_dims = lat.variable.dims
        assert lon.variable.dims == self.grid_dims, (lon.variable.dims, self.grid_dims)
        assert set(self.variable_dims) == set(self.grid_dims), (self.variable_dims, self.grid_dims)

    @cached_property
    def grid_points(self):

        assert 1 <= len(self.variable_dims) <= 2

        if len(self.variable_dims) == 1:
            return self.lat.variable.values.flatten(), self.lon.variable.values.flatten()

        if len(self.variable_dims) == 2 and self.variable_dims == self.grid_dims:
            return self.lat.variable.values.flatten(), self.lon.variable.values.flatten()

        LOG.warning(
            "UnstructuredGrid: variable indexing %s does not match grid indexing %s", self.variable_dims, self.grid_dims
        )

        lat = self.lat.variable.values.transpose().flatten()
        lon = self.lon.variable.values.transpose().flatten()

        return lat, lon


class ProjectionGrid(XYGrid):
    def __init__(self, x, y, projection):
        super().__init__(x, y)
        self.projection = projection

    def transformer(self):
        from pyproj import CRS
        from pyproj import Transformer

        if isinstance(self.projection, dict):
            data_crs = CRS.from_cf(self.projection)
        else:
            data_crs = self.projection

        wgs84_crs = CRS.from_epsg(4326)  # WGS84

        return Transformer.from_crs(data_crs, wgs84_crs, always_xy=True)


class MeshProjectionGrid(ProjectionGrid):

    @cached_property
    def grid_points(self):

        transformer = self.transformer()
        xv, yv = np.meshgrid(self.x.variable.values, self.y.variable.values)  # , indexing="ij")
        lon, lat = transformer.transform(xv, yv)
        return lat.flatten(), lon.flatten()


class UnstructuredProjectionGrid(XYGrid):
    @cached_property
    def grid_points(self):
        assert False, "Not implemented"

        # lat, lon = transformer.transform(
        #      self.y.variable.values.flatten(),
        #     self.x.variable.values.flatten(),

        # )

        # lat = lat[::len(lat)//100]
        # lon = lon[::len(lon)//100]

        # print(len(lat), len(lon))

        # return np.meshgrid(lat, lon)
