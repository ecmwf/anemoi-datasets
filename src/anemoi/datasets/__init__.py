# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import List

from .data import MissingDateError
from .data import add_dataset_path
from .data import add_named_dataset
from .data import list_dataset_names
from .data import open_dataset

try:
    # NOTE: the `_version.py` file must not be present in the git repository
    #   as it is generated by setuptools at install time
    from ._version import __version__  # type: ignore
except ImportError:  # pragma: no cover
    # Local copy or not installed with setuptools
    __version__ = "999"

__all__: List[str] = [
    "add_dataset_path",
    "add_named_dataset",
    "list_dataset_names",
    "MissingDateError",
    "open_dataset",
    "__version__",
]
