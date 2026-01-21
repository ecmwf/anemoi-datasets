# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime

import numpy as np
import torch


class AnnotatedTorchTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, meta=None, *args, **kwargs):
        # Create the tensor instance
        obj = super().__new__(cls, data, *args, **kwargs)
        obj.meta = meta
        return obj

    def __init__(self, data, meta=None, *args, **kwargs):
        # super().__init__(data, *args, **kwargs)
        self.meta = meta

    @property
    def dates(self) -> np.ndarray:
        """Array of dates associated with the data."""
        return self.meta.dates

    @property
    def latitudes(self) -> torch.Tensor:
        """Array of latitudes associated with the data."""
        return torch.tensor(self.meta.latitudes, device=self.device)

    @property
    def longitudes(self) -> torch.Tensor:
        """Array of longitudes associated with the data."""
        return torch.tensor(self.meta.longitudes, device=self.device)

    @property
    def timedeltas(self) -> torch.Tensor:
        """Array of time deltas associated with the data."""
        return torch.tensor(self.meta.timedeltas, device=self.device)

    @property
    def reference_date(self) -> datetime.datetime:
        """The reference date for the data."""
        return self.meta.reference_date

    @property
    def boundaries(self) -> list[slice]:
        """The boundaries for the data."""
        return self.meta.boundaries

    @property
    def index(self) -> list[slice]:
        """The index for the data."""
        return self.meta.index
