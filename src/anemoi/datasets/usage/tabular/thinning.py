# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..dataset import Dataset
from ..forwards import Forwards
from ..mixins.thinning import ThinningMixin

LOG = logging.getLogger(__name__)


class Thinning(ThinningMixin, Forwards):
    """A class to represent a thinned dataset."""

    def __init__(self, forward: Dataset, thinning: int | float | None, method: str) -> None:
        """Initialize the Thinning class.

        Parameters
        ----------
        forward : Dataset
            The dataset to be thinned.
        thinning : Optional[int]
            The thinning factor.
        method : str
            The thinning method.
        """

        # Set the attributes before calling the thinner property in the mixin

        self.thinning = thinning
        self.method = method
        self.grid_shape = None

        super().__init__(forward)

    def __getitem__(self, n):
        data = super().__getitem__(n)
        latitudes = data[:, 3]
        longitudes = data[:, 4]
        mask = self.thinner.mask(latitudes, longitudes)
        return data[mask]
