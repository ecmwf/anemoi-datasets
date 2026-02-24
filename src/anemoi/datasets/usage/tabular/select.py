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

from numpy.typing import NDArray

from anemoi.datasets.usage.dataset import FullIndex

from ..common.select import SelectBase

LOG = logging.getLogger(__name__)


class Select(SelectBase):
    """Class to select a subset of variables from a dataset."""

    def __getitem__(self, n: FullIndex) -> NDArray[Any]:
        """Get an item from the dataset.

        Parameters
        ----------
        n : FullIndex
            The index to retrieve.

        Returns
        -------
        NDArray[Any]
            The retrieved data.
        """

        tensor = self.dataset[n]
        return tensor[:, self.indices]
