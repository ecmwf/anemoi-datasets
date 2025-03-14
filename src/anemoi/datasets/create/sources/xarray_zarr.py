# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import List

import earthkit.data as ekd

from .xarray import load_many


def execute(context: Any, dates: List[str], url: str, *args: Any, **kwargs: Any) -> ekd.FieldList:

    return load_many("🇿", context, url, *args, **kwargs).execute(dates)
