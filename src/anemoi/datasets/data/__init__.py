# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from .misc import _open_dataset
from .misc import add_dataset_path
from .misc import add_named_dataset

LOG = logging.getLogger(__name__)

__all__ = [
    "open_dataset",
    "MissingDateError",
    "add_dataset_path",
    "add_named_dataset",
]


class MissingDateError(Exception):
    pass


def open_dataset(*args, **kwargs):
    ds = _open_dataset(*args, **kwargs)
    ds = ds.mutate()
    ds.arguments = {"args": args, "kwargs": kwargs}
    ds._check()
    return ds


def list_dataset_names(*args, **kwargs):
    ds = _open_dataset(*args, **kwargs)
    names = set()
    ds.get_dataset_names(names)
    return sorted(names)
