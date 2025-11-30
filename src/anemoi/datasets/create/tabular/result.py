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

from anemoi.datasets.create.input.result import Result

LOG = logging.getLogger(__name__)


class TabularResult(Result):
    """Class to represent the result of an action in the dataset creation process."""

    def __init__(self, context: Any, argument: Any, datasource: Any) -> None:

        pass
