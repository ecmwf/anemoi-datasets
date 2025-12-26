# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import BeforeValidator

from ..config import OutputSpecs

LOG = logging.getLogger(__name__)


def validate_order_by(v):
    if isinstance(v, (list, tuple)):
        return {k: "ascending" for k in v}
    return dict(v)


class Output(BaseModel, OutputSpecs):
    order_by: Annotated[list[str] | dict[str, str], BeforeValidator(validate_order_by)] = validate_order_by(
        ["valid_datetime", "param_level", "number"]
    )

    flatten_grid: bool = True
    remapping: dict[str, Any] | None = None
