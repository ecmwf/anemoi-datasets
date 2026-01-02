# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("crop")
class Crop(Filter):

    def __init__(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
    ):

        self.north = north
        self.south = south
        self.east = east
        self.west = west

    def forward(self, data):
        LOG.info(f"Cropping to N:{self.north}, S:{self.south}, E:{self.east}, W:{self.west}")
        LOG.info(f"Parameter type: {type(data)}")
        return data
