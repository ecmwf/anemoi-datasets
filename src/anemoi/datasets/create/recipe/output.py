# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from ..config import OutputSpecs

LOG = logging.getLogger(__name__)


# def validate_order_by(v):
#     if isinstance(v, (list, tuple)):
#         return {k: "ascending" for k in v}
#     return dict(v)


class Output(BaseModel, OutputSpecs):
    dtype: str = "float32"
    flatten_grid: bool = True

    order_by: list[str] = Field(default_factory=lambda: ["valid_datetime", "param_level", "number"])

    remapping: dict[str, Any] = Field(default_factory=lambda: {"param_level": "{param}_{levelist}"})
    chunking: dict[str, int] = Field(default_factory=lambda: {"dates": 1, "ensembles": 1})
