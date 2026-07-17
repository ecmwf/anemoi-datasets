# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Leaf pydantic building blocks shared by the recipe schemas and the source schemas.

``Frequency`` and ``Steps`` are used both by the recipe models
(``create/recipe/dates.py``, which re-exports them) and by per-source
validation schemas such as the accumulate archive description.  They live
in this import-light module — importing nothing from the ``recipe`` or
``sources`` packages — so that source modules can use them without
creating an import cycle with the recipe package (whose ``Action`` union
imports every source at import time).
"""

from __future__ import annotations

import datetime
from functools import cached_property
from typing import Annotated

from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import ConfigDict
from pydantic import PlainSerializer
from pydantic import model_validator

# A datetime.timedelta that accepts frequency strings (e.g. "6h") on input
# and serialises back to the same short form (e.g. "6h") rather than pydantic's
# default ISO 8601 duration (e.g. "PT6H").
Frequency = Annotated[
    datetime.timedelta,
    BeforeValidator(frequency_to_timedelta),
    PlainSerializer(frequency_to_string, return_type=str, when_used="json"),
]


class Steps(BaseModel):
    """Forecast lead times for the ``trajectories`` layout.

    Models a regular range of steps via ``start`` / ``end`` / ``frequency``
    (all parsed as :class:`datetime.timedelta`).  Exposes the same iteration and
    ``numpy`` array interface the trajectories pipeline relies on.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: Frequency
    end: Frequency
    frequency: Frequency

    @model_validator(mode="after")
    def _check_range(self) -> "Steps":
        if self.frequency <= datetime.timedelta(0):
            raise ValueError(f"'steps.frequency' must be positive, got {self.frequency}")
        if self.end < self.start:
            raise ValueError(f"'steps.end' ({self.end}) must be >= 'steps.start' ({self.start})")
        if (self.end - self.start) % self.frequency != datetime.timedelta(0):
            raise ValueError(
                f"'steps.frequency' ({frequency_to_string(self.frequency)}) must divide "
                f"'steps.end' - 'steps.start' "
                f"({frequency_to_string(self.start)} to {frequency_to_string(self.end)})"
            )
        # The pipeline (MARS step requests, per-field placement) assumes
        # whole-hour steps throughout.
        for name in ("start", "end", "frequency"):
            value = getattr(self, name)
            if value.total_seconds() % 3600:
                raise ValueError(
                    f"'steps.{name}' must be a whole number of hours, " f"got {frequency_to_string(value)}"
                )
        return self

    @cached_property
    def values(self):
        import numpy as np

        return np.arange(self.start, self.end + self.frequency, self.frequency)

    def dump(self, dumper):
        return dumper.steps(self.start, self.end, self.frequency)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __array__(self, dtype=None, copy=None):
        arr = self.values if dtype is None else self.values.astype(dtype)
        return arr.copy() if copy else arr
