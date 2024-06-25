# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from .debug import Node
from .forwards import GivenAxis
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class Ensemble(GivenAxis):
    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])


def ensemble_factory(args, kwargs):
    if "grids" in kwargs:
        raise NotImplementedError("Cannot use both 'ensemble' and 'grids'")

    ensemble = kwargs.pop("ensemble")
    axis = kwargs.pop("axis", 2)
    assert len(args) == 0
    assert isinstance(ensemble, (list, tuple))

    datasets = [_open(e) for e in ensemble]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Ensemble(datasets, axis=axis)._subset(**kwargs)
