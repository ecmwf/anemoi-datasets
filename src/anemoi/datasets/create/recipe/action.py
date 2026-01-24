# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import datetime
import logging
from functools import cache
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Union

from anemoi.utils.dates import DateTimes
from anemoi.utils.dates import as_datetime
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta
from anemoi.utils.hindcasts import HindcastDatesTimes
from anemoi.utils.humanize import print_dates
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag
from pydantic import create_model
from pydantic_core import PydanticCustomError

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


@cache
def _factories():
    from anemoi.transform.filters import filter_registry as transform_filter_registry
    from anemoi.transform.sources import source_registry as transform_source_registry

    from anemoi.datasets.create.sources import source_registry as dataset_source_registry

    union = []

    factories = {}

    factories.update(transform_filter_registry.factories)
    factories.update(transform_source_registry.factories)
    factories.update(dataset_source_registry.factories)

    return {name.replace("-", "_"): klass for name, klass in factories.items()}


@cache
def _schemas():
    from anemoi.datasets.create.sources import source_registry as dataset_source_registry

    union = []

    for name, klass in _factories().items():
        schema = getattr(klass, "schema", None)
        if schema is None:
            schema = dict
        name = name.replace("-", "_")
        model = create_model(name, **{name: (schema, ...)}, __base__=Function)
        union.append(Annotated[model, Tag(name)])

    union.extend(
        [
            Annotated[Pipe, Tag("pipe")],
            Annotated[Join, Tag("join")],
        ]
    )

    return tuple(union)


def _step_discriminator(options: Any) -> str:

    assert len(options) == 1, options

    verb = list(options.keys())[0]
    verb = verb.replace("-", "_")

    # This will give us a much more readable error message than the default pydantic exception

    if verb not in ("pipe", "join"):
        if verb not in _factories():
            raise PydanticCustomError("unknown-name", f"Unknown source or filter: '{verb}'")

    return verb


Step = Annotated[Union[_schemas()], Discriminator(_step_discriminator)]


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
