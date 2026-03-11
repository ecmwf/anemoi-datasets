# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import numpy as np

from anemoi.datasets.usage.dataset import Dataset
from anemoi.datasets.usage.debug import Node
from anemoi.datasets.usage.forwards import Forwards

LOG = logging.getLogger(__name__)


class Cropping(Forwards):
    """A class to represent a cropped dataset."""

    def __init__(self, forward: Dataset, area: Dataset | tuple[float, float, float, float]) -> None:
        """Initialize the Cropping class.

        Parameters
        ----------
        forward : Dataset
            The dataset to be cropped.
        area : Union[Dataset, Tuple[float, float, float, float]]
            The cropping area.
        """
        from anemoi.datasets import open_dataset

        area = area if isinstance(area, (list, tuple)) else open_dataset(area)

        if isinstance(area, Dataset):
            north = np.amax(area.latitudes)
            south = np.amin(area.latitudes)
            east = np.amax(area.longitudes)
            west = np.amin(area.longitudes)
            area = (north, west, south, east)

        self.area = area

        super().__init__(forward)

    def __getitem__(self, n):
        result = self.forward[n]
        north, west, south, east = self.area
        latitudes = result[:, 1]
        longitudes = result[:, 2]

        north, west, south, east = self.area

        west = west % 360
        east = east % 360
        if west > east:
            # Crossing the meridian
            mask = (latitudes <= north) & (latitudes >= south) & ((longitudes >= west) | (longitudes <= east))
        else:
            # Create mask for points inside the area
            mask = (latitudes <= north) & (latitudes >= south) & (longitudes >= west) & (longitudes <= east)

        return result[mask]

    def tree(self) -> Node:
        """Get the tree representation of the dataset.

        Returns
        -------
        Node
            The tree representation of the dataset.
        """
        return Node(self, [self.forward.tree()], area=self.area)

    def forwards_subclass_metadata_specific(self) -> dict[str, Any]:
        """Get the metadata specific to the Cropping subclass.

        Returns
        -------
        Dict[str, Any]
            The metadata specific to the Cropping subclass.
        """
        return dict(area=self.area)

    def origin_transformation(self, variable, origins):
        return {"name": "cropping", "config": dict(area=self.area)}
