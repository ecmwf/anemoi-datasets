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

from .dataset import Dataset
from .dataset import FullIndex
from .dataset import Shape
from .dataset import TupleIndex
from .misc import _open_dataset
from .misc import add_dataset_path
from .misc import add_named_dataset

LOG = logging.getLogger(__name__)

__all__ = [
    "open_dataset",
    "MissingDateError",
    "add_dataset_path",
    "add_named_dataset",
    "FullIndex",
    "TupleIndex",
    "Shape",
]


class MissingDateError(Exception):
    pass


def _convert(x: Any) -> Any:

    if isinstance(x, list):
        return [_convert(a) for a in x]

    if isinstance(x, tuple):
        return tuple(_convert(a) for a in x)

    if isinstance(x, dict):
        return {k: _convert(v) for k, v in x.items()}

    if x.__class__.__name__ in ("DictConfig", "ListConfig"):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(x, resolve=True)

    return x


def open_dataset(*args: Any, **kwargs: Any) -> Dataset:

    # That will get rid of OmegaConf objects

    args, kwargs = _convert(args), _convert(kwargs)

    ds = _open_dataset(*args, **kwargs)
    ds = ds.mutate()
    ds.arguments = {"args": args, "kwargs": kwargs}
    ds._check()
    return ds


def list_dataset_names(*args: Any, **kwargs: Any) -> list[str]:
    ds = _open_dataset(*args, **kwargs)
    names: set[str] = set()
    ds.get_dataset_names(names)
    return sorted(names)
