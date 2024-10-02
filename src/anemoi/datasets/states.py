# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
import numpy as np


LOG = logging.getLogger(__name__)


def str_(t):
    """Not needed, but useful for debugging"""
    import numpy as np

    if isinstance(t, (list, tuple)):
        return "[" + " , ".join(str_(e) for e in t) + "]"
    if isinstance(t, np.ndarray):
        return "np:" + str(t.shape).replace(" ", "").replace(",", "-").replace("(", "").replace(")", "")
    if isinstance(t, dict):
        return "{" + " , ".join(f"{k}: {str_(v)}" for k, v in t.items()) + "}"
    try:
        from torch import Tensor

        if isinstance(t, Tensor):
            return "tor:" + str(tuple(t.size())).replace(" ", "").replace(",", "-").replace("(", "").replace(")", "")
    except ImportError:
        pass
    return str(t)


class AnemoiSample:
    def __init__(self, list_of_list_of_arrays):
        def cast_to_state(v):
            if isinstance(v, AnemoiState):
                return v
            return AnemoiState(v)

        self._states = tuple(cast_to_state(_) for _ in list_of_list_of_arrays)

    def __iter__(self):
        return iter(self._states)

    def __str__(self):
        return f"AnemoiSample({str_(self._states)})"

    @property
    def dtype(self):
        return self._states[0].dtype

    def to(self, device):
        return self.__class__([s.to(device) for s in self])

    def numpy_to_torch(self):
        return self.__class__([v.numpy_to_torch() for v in self])

    def as_tuple_of_tuples(self):
        return tuple(v.as_tuple() for v in self)

    def as_tuple_of_dicts(self, keys=None):
        return tuple(v.as_dict(keys) for v in self)


class TrainingAnemoiSample(AnemoiSample):
    pass


class InferenceAnemoiSample(AnemoiSample):
    pass


class AnemoiState:
    def __init__(self, arrays):
        self.arrays = arrays
        self.lenghts = [v.size for v in arrays]

        # check all arrays have the same type
        for a in self.arrays:
            assert isinstance(a, self._type), (type(a), self._type)

    def numpy_to_torch(self):
        import torch

        return self.__class__([torch.from_numpy(v) for v in self.arrays])

    @property
    def _type(self):
        if self.arrays:
            return type(self.arrays[0])
        return None

    def __getitem__(self, tupl):
        if len(tupl) == 1:
            return self.arrays[tupl[0]]
        assert len(tupl) == 2
        i, j = tupl
        return self.arrays[i][j]

    def __setitem__(self, tupl, value):
        # if len(tupl) == 1:
        #     self.arrays[tupl[0]] = value
        assert len(tupl) == 2
        i, j = tupl
        self.arrays[i][j] = value

    @property
    def size(self):
        return sum(v.size for v in self.arrays)

    def flatten(self):
        return np.concatenate([v.flatten() for v in self.arrays])

    def map(self, f):
        return AnemoiState([f(v) for v in self.arrays])

    @cached_property
    def dtype(self):
        assert all(v.dtype == self.arrays[0].dtype for v in self.arrays)
        return self.arrays[0].dtype

    def __repr__(self):
        return f"AnemoiState({str_(self.arrays)})"

    def as_list(self):
        return list(self.arrays)

    def as_tuple(self):
        return tuple(self.arrays)

    def as_dict(self, keys=None):
        assert keys is not None, f"Using a list of keys from the config is not implemented yet"
        # todo here: get the list of keys.
        assert len(keys) == len(self.arrays), (len(keys), len(self.arrays))
        return {k: v for k, v in zip(keys, self.arrays)}

    def to(self, device):
        return self.__class__([v.to(device) for v in self.arrays])