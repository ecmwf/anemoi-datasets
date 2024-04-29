# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .data import MissingDateError
from .data import add_dataset_path
from .data import add_named_dataset
from .data import open_dataset

__version__ = "0.1.4"

__all__ = [
    "open_dataset",
    "MissingDateError",
    "add_dataset_path",
    "add_named_dataset",
]
