# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
from typing import Any

import numpy as np


def extract_dates_from_results(x: Any) -> str:

    # Used to print dates in debug statements (e.g., in bisect.py, view.py)

    if x is None:
        return "None"

    if isinstance(x, np.int64):
        return extract_dates_from_results(int(x))

    if isinstance(x, np.ndarray):
        return extract_dates_from_results(tuple(int(i) for i in x))

    if isinstance(x, tuple):
        return (extract_dates_from_results(int(x[0])),) + x[1:]

    if isinstance(x, datetime.datetime):
        return x.isoformat()

    assert isinstance(x, int), (x, type(x))

    return datetime.datetime.fromtimestamp(x).isoformat()
