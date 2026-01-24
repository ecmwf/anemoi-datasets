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
    # @staticmethod
    def __new__(cls, data, anemoi_annotation=None, *args, **kwargs):
        # Create the tensor instance
        obj = super().__new__(cls, data, *args, **kwargs)
        obj._anemoi_annotation = anemoi_annotation
        return obj

    def __init__(self, data, anemoi_annotation=None, *args, **kwargs):
        # super().__init__(data, *args, **kwargs)
        self._anemoi_annotation = anemoi_annotation

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Run the actual torch operation
        ret = super().__torch_function__(func, types, args, kwargs)

        # Try to find the metadata from the original tensor(s)
        # and propagate it to the result
        _anemoi_annotation = None
        for arg in args:
            if isinstance(arg, AnnotatedTorchTensor):
                _anemoi_annotation = arg._anemoi_annotation
                break

        # If the result is a tensor, wrap it back into AnnotatedTorchTensor
        if isinstance(ret, torch.Tensor) and not isinstance(ret, AnnotatedTorchTensor):
            ret = AnnotatedTorchTensor(ret, anemoi_annotation=_anemoi_annotation)

        return ret

    # This tells multiprocessing how to rebuild the object
    def __reduce_ex__(self, protocol):
        print("REDUCE EX CALLED")
        # Get the standard tensor reduction
        reduce_value = super().__reduce_ex__(protocol)
        func, args, state, list_iter, dict_iter = reduce_value

        # Add our metadata to the 'state' part of the pickle
        new_state = (state, {"_anemoi_annotation": self._anemoi_annotation})
        return (func, args, new_state, list_iter, dict_iter)

    def __setstate__(self, state):
        print("SETSTATE CALLED")
        # Unpack the standard state and our custom metadata
        torch_state, custom_dict = state
        super().__setstate__(torch_state)
        self._anemoi_annotation = custom_dict["_anemoi_annotation"]

    @property
    def dates(self) -> np.ndarray:
        """Array of dates associated with the data."""
        return self._anemoi_annotation.dates

    @property
    def latitudes(self) -> torch.Tensor:
        """Array of latitudes associated with the data."""
        return torch.tensor(self._anemoi_annotation.latitudes, device=self.device)

    @property
    def longitudes(self) -> torch.Tensor:
        """Array of longitudes associated with the data."""
        return torch.tensor(self._anemoi_annotation.longitudes, device=self.device)

    @property
    def timedeltas(self) -> torch.Tensor:
        """Array of time deltas associated with the data."""
        return torch.tensor(self._anemoi_annotation.timedeltas, device=self.device)

    @property
    def reference_date(self) -> datetime.datetime:
        """The reference date for the data."""
        return self._anemoi_annotation.reference_date

    @property
    def reference_dates(self) -> datetime.datetime:
        """The reference date for the data."""
        return self._anemoi_annotation.reference_dates

    @property
    def boundaries(self) -> list[slice]:
        """The boundaries for the data."""
        return self._anemoi_annotation.boundaries

    @property
    def index(self) -> list[slice]:
        """The index for the data."""
        return self._anemoi_annotation.index
