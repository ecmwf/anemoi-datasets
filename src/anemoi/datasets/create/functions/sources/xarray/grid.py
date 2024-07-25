# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import numpy as np


class Grid:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    @property
    def latitudes(self):
        return self.grid_points()[0]

    @property
    def longitudes(self):
        return self.grid_points()[1]


class MeshedGrid(Grid):
    _cache = None

    def grid_points(self):
        if self._cache is not None:
            return self._cache
        lat = self.lat.variable.values
        lon = self.lon.variable.values

        lat, lon = np.meshgrid(lat, lon)
        self._cache = (lat.flatten(), lon.flatten())
        return self._cache


class UnstructuredGrid(Grid):
    def grid_points(self):
        lat = self.lat.variable.values.flatten()
        lon = self.lon.variable.values.flatten()
        return lat, lon
