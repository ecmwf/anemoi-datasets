# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

import logging
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from pydantic import BaseModel, Field
from pydantic import BeforeValidator
from pydantic import Field
from pydantic_core import PydanticCustomError
from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
import datetime
from anemoi.utils.dates import DateTimes
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.hindcasts import HindcastDatesTimes
from anemoi.utils.humanize import print_dates


LOG = logging.getLogger(__name__)


class BaseAction(BaseModel):
    pass


class Pipe(BaseAction):
    pipe: list[Action] = []


class Join(BaseAction):
    join: list[Action] = []


class Concat(BaseAction):
    concat: tuple[dict, Action]


class Function(BaseAction):
    pass


def _step_discriminator(options: Any) -> str:

    from ..input.action import sources_and_filters_factories

    BUILTINS = ("pipe", "join")

    assert len(options) == 1, options

    verb = list(options.keys())[0]

    if verb in BUILTINS:
        return verb

    for name, klass in sources_and_filters_factories().items():
        pass

    return "function"


Step = Annotated[
    Union[
        Annotated[Pipe, Tag("pipe")],
        Annotated[Join, Tag("join")],
        Annotated[dict, Tag("function")],
    ],
    Discriminator(_step_discriminator),
]


def _action_discriminator(options: dict) -> str:
    if len(options) == 2 and "dates" in options:
        return "concat"
    return "step"


Action = Annotated[
    Union[
        Annotated[Step, Tag("step")],
        Annotated[Concat, Tag("concat")],
    ],
    Discriminator(_action_discriminator),
]
