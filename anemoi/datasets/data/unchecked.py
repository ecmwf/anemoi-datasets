# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property
from functools import wraps

from .concat import ConcatMixin
from .debug import Node
from .forewards import Combined
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class check:

    def __init__(self, check):
        self.check = check

    def __call__(self, method):
        name = method.__name__
        check = self.check

        @wraps(method)
        def wrapper(obj):
            """
            This is a decorator that checks the compatibility of the datasets
            before calling the method. If the datasets are compatible, it
            will return the result of the method, otherwise it will raise an
            exception.
            """

            for d in obj.datasets[1:]:
                getattr(obj, check)(obj.datasets[0], d)

            return getattr(Combined, name).__get__(obj)

        return wrapper


class Unchecked(Combined):

    def tree(self):
        return Node(self, [d.tree() for d in self.datasets])

    def _subset(self, **kwargs):
        assert not kwargs
        return self

    def check_compatibility(self, d1, d2):
        pass

    ###########################################
    @property
    @check("check_same_dates")
    def dates(self):
        pass

    @property
    @check("check_same_resolution")
    def resolution(self):
        pass

    @property
    def field_shape(self):
        raise NotImplementedError()

    @property
    @check("check_same_frequency")
    def frequency(self):
        raise NotImplementedError()

    @property
    @check("check_same_grid")
    def latitudes(self):
        raise NotImplementedError()

    @property
    @check("check_same_grid")
    def longitudes(self):
        raise NotImplementedError()

    @property
    @check("check_same_variables")
    def name_to_index(self):
        raise NotImplementedError()

    @property
    @check("check_same_variables")
    def variables(self):
        raise NotImplementedError()

    @property
    @check("check_same_variables")
    def statistics(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def grids(self):
        raise NotImplementedError()


class Zip(Unchecked):

    def __len__(self):
        return min(len(d) for d in self.datasets)

    def __getitem__(self, n):
        return tuple(d[n] for d in self.datasets)

    @cached_property
    def missing(self):
        result = set()
        for d in self.datasets:
            result = result | d.missing
        return result


class Chain(ConcatMixin, Unchecked):
    """
    Same as Concat, but with no checks
    """

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, n):
        return tuple(d[n] for d in self.datasets)

    @property
    def dates(self):
        raise NotImplementedError()


def zip_factory(args, kwargs, zarr_root):

    zip = kwargs.pop("zip")
    assert len(args) == 0
    assert isinstance(zip, (list, tuple))

    datasets = [_open(e, zarr_root) for e in zip]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Zip(datasets)._subset(**kwargs)


def chain_factory(args, kwargs, zarr_root):

    chain = kwargs.pop("chain")
    assert len(args) == 0
    assert isinstance(chain, (list, tuple))

    datasets = [_open(e, zarr_root) for e in chain]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Chain(datasets)._subset(**kwargs)
