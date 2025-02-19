# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging
from functools import cached_property
from functools import wraps
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy as np
from numpy.typing import NDArray

from .concat import ConcatMixin
from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .debug import Node
from .forwards import Combined
from .misc import _auto_adjust
from .misc import _open

LOG = logging.getLogger(__name__)


class check:
    def __init__(self, check: str) -> None:
        self.check = check

    def __call__(self, method: callable) -> callable:
        name = method.__name__
        check = self.check

        @wraps(method)
        def wrapper(obj: "Unchecked") -> Any:
            """This is a decorator that checks the compatibility of the datasets before calling the method.
            If the datasets are compatible, it will return the result of the method,
            otherwise it will raise an exception."""

            for d in obj.datasets[1:]:
                getattr(obj, check)(obj.datasets[0], d)

            return getattr(Combined, name).__get__(obj)

        return wrapper


class Unchecked(Combined):
    def tree(self) -> Node:
        return Node(self, [d.tree() for d in self.datasets])

    def _subset(self, **kwargs: dict) -> "Unchecked":
        assert not kwargs
        return self

    def check_compatibility(self, d1: Dataset, d2: Dataset) -> None:
        pass

    ###########################################
    @property
    @check("check_same_dates")
    def dates(self) -> NDArray[np.datetime64]:
        pass

    @property
    @check("check_same_resolution")
    def resolution(self) -> Any:
        pass

    @property
    def field_shape(self) -> tuple:
        raise NotImplementedError()

    @property
    @check("check_same_frequency")
    def frequency(self) -> datetime.timedelta:
        raise NotImplementedError()

    @property
    @check("check_same_grid")
    def latitudes(self) -> NDArray[Any]:
        raise NotImplementedError()

    @property
    @check("check_same_grid")
    def longitudes(self) -> NDArray[Any]:
        raise NotImplementedError()

    @check("check_same_variables")
    @property
    def name_to_index(self) -> Dict[str, int]:
        raise NotImplementedError()

    @check("check_same_variables")
    @property
    def variables(self) -> List[str]:
        raise NotImplementedError()

    @check("check_same_variables")
    @property
    def variables_metadata(self) -> dict:
        raise NotImplementedError()

    @check("check_same_variables")
    @property
    def statistics(self) -> Dict[str, NDArray[Any]]:
        raise NotImplementedError()

    @check("check_same_variables")
    def statistics_tendencies(self, delta: Optional[datetime.timedelta] = None) -> Dict[str, NDArray[Any]]:
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        raise NotImplementedError()

    @cached_property
    def missing(self) -> Set[int]:
        result: Set[int] = set()
        for d in self.datasets:
            result = result | d.missing
        return result


class Chain(ConcatMixin, Unchecked):
    """Same as Concat, but with no checks"""

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, n: FullIndex) -> tuple:
        return tuple(d[n] for d in self.datasets)

    @property
    def dates(self) -> NDArray[np.datetime64]:
        raise NotImplementedError()

    def dataset_metadata(self) -> dict:
        return {"multiple": [d.dataset_metadata() for d in self.datasets]}


def chain_factory(args: tuple, kwargs: dict) -> Chain:
    chain = kwargs.pop("chain")
    assert len(args) == 0
    assert isinstance(chain, (list, tuple))

    datasets = [_open(e) for e in chain]
    datasets, kwargs = _auto_adjust(datasets, kwargs)

    return Chain(datasets)._subset(**kwargs)
