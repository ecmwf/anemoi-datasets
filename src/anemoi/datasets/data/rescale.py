# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from functools import cached_property

import numpy as np

from .debug import Node
from .debug import debug_indexing
from .forwards import Forwards
from .indexing import apply_index_to_slices_changes
from .indexing import expand_list_indexing
from .indexing import index_to_slices
from .indexing import update_tuple

LOG = logging.getLogger(__name__)


def make_rescale(variable, rescale):

    if isinstance(rescale, (tuple, list)):

        assert len(rescale) == 2, rescale

        if isinstance(rescale[0], (int, float)):
            return rescale

        from cfunits import Units

        u0 = Units(rescale[0])
        u1 = Units(rescale[1])

        x1, x2 = 0.0, 1.0
        y1, y2 = Units.conform([x1, x2], u0, u1)

        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        return a, b

        return rescale

    if isinstance(rescale, dict):
        assert "scale" in rescale, rescale
        assert "offset" in rescale, rescale
        return rescale["scale"], rescale["offset"]

    assert False


class Rescale(Forwards):
    def __init__(self, dataset, rescale):
        super().__init__(dataset)
        for n in rescale:
            assert n in dataset.variables, n

        variables = dataset.variables
        print(variables)

        self._a = np.ones(len(variables))
        self._b = np.zeros(len(variables))

        for i, v in enumerate(variables):
            if v in rescale:
                print(i)
                self._a[i], self._b[i] = make_rescale(v, rescale[v])

        self._a = self._a[np.newaxis, :, np.newaxis, np.newaxis]
        self._b = self._b[np.newaxis, :, np.newaxis, np.newaxis]

        self.rescale = rescale

    @property
    def variables(self):
        return self._variables

    def tree(self):
        return Node(self, [self.forward.tree()], rescale=self.rescale)

    def subclass_metadata_specific(self):
        return dict(rescale=self.rescale)

    @debug_indexing
    @expand_list_indexing
    def _get_tuple(self, index):
        index, changes = index_to_slices(index, self.shape)
        index, previous = update_tuple(index, 1, slice(None))
        result = self.forward[index]
        result = result * self._a + self._b
        result = result[:, previous]
        result = apply_index_to_slices_changes(result, changes)
        return result

    @debug_indexing
    def __get_slice_(self, n):
        data = self.forward[n]
        return data * self._a + self._b

    @debug_indexing
    def __getitem__(self, n):

        if isinstance(n, tuple):
            return self._get_tuple(n)

        if isinstance(n, slice):
            return self.__get_slice_(n)

        data = self.forward[n]

        return data * self._a[0] + self._b[0]

    @cached_property
    def statistics(self):
        warnings.warn("Statistics are not rescaled")
        return self.forward.statistics

    def statistics_tendencies(self, delta=None):
        warnings.warn("Statistics are not rescaled")
        return self.forward.statistics_tendencies(delta)
