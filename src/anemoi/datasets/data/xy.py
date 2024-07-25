# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

from .debug import Node
from .forwards import Combined
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class ZipBase(Combined):

    def swap_with_parent(self, parent):
        new_parents = [parent.clone(ds) for ds in self.datasets]
        return self.clone(new_parents)

    def clone(self, datasets):
        return self.__class__(datasets)

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])

    def __len__(self):
        return min(len(d) for d in self.datasets)

    def __getitem__(self, n):
        return tuple(d[n] for d in self.datasets)

    def check_same_resolution(self, d1, d2):
        pass

    def check_same_grid(self, d1, d2):
        pass

    def check_same_variables(self, d1, d2):
        pass

    @cached_property
    def missing(self):
        result = set()
        for d in self.datasets:
            result = result | d.missing
        return result

    @property
    def shape(self):
        return tuple(d.shape for d in self.datasets)

    @property
    def field_shape(self):
        return tuple(d.shape for d in self.datasets)

    @property
    def latitudes(self):
        return tuple(d.latitudes for d in self.datasets)

    @property
    def longitudes(self):
        return tuple(d.longitudes for d in self.datasets)

    @property
    def dtype(self):
        return tuple(d.dtype for d in self.datasets)

    @property
    def grids(self):
        return tuple(d.grids for d in self.datasets)

    @property
    def statistics(self):
        return tuple(d.statistics for d in self.datasets)

    @property
    def resolution(self):
        return tuple(d.resolution for d in self.datasets)

    @property
    def name_to_index(self):
        return tuple(d.name_to_index for d in self.datasets)


class Zip(ZipBase):
    pass


class XY(ZipBase):
    pass


def xy_factory(args, kwargs):

    if "xy" in kwargs:
        xy = kwargs.pop("xy")
    else:
        xy = [kwargs.pop("x"), kwargs.pop("y")]

    assert len(args) == 0
    assert isinstance(xy, (list, tuple))

    datasets = [_open(e) for e in xy]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    assert len(datasets) == 2

    return XY(datasets)._subset(**kwargs)


def zip_factory(args, kwargs):

    zip = kwargs.pop("zip")
    assert len(args) == 0
    assert isinstance(zip, (list, tuple))

    datasets = [_open(e) for e in zip]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Zip(datasets)._subset(**kwargs)
