# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``AccumulateSchema`` — the recipe-facing validation model for ``accumulate``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from anemoi.datasets.create.time_schemas import Frequency

from .normalise import normalise_from
from .union import From
from .union import FromBare
from .union import FromLookupTable
from .union import FromTrajectories


def _hyphen_alias(name: str) -> str:
    return name.replace("_", "-")


class AccumulateSchema(BaseModel):
    """Validation schema for the ``accumulate`` source in recipes.

    ``period:`` is what the recipe wants; ``from:`` is what the source data
    is.  ``from:`` is optional: when omitted (``from_`` is ``None``) the
    source data is recognised from the source at build time — there is no
    ``auto`` value to write.  Every accepted
    spelling of the description is folded into ``from:`` by
    :func:`normalise_from` during validation, so afterwards ``self.from_``
    is the only description that matters.  Whether
    ``from.base_dates``/``from.steps`` are required or forbidden depends
    on the output layout, so that rule lives in the ``Recipe`` model —
    the only place that knows the layout.
    """

    model_config = ConfigDict(
        alias_generator=_hyphen_alias,
        populate_by_name=True,
        extra="forbid",
    )

    period: Frequency
    source: dict[str, Any]

    from_: From | None = Field(default=None, alias="from")

    patch: list[str] | None = None
    group_by: dict[str, Any] | None = None

    # Deprecated spellings, kept for one release.  They are folded into
    # `from_` during validation (so they are always None afterwards) and
    # excluded from dumps.
    accumulation: str | None = Field(default=None, exclude=True)
    covering: Any = None
    availability: Any = Field(default=None, exclude=True)

    @property
    def from_kind(self) -> str | None:
        """A short label for the ``from:`` description in use, or ``None`` when omitted."""
        if self.from_ is None:
            return None
        if isinstance(self.from_, FromTrajectories):
            return "trajectories (base_dates × steps)"
        if isinstance(self.from_, FromLookupTable):
            return "lookup-table"
        if isinstance(self.from_, FromBare):
            return "bare (accumulation only)"
        return None

    @model_validator(mode="before")
    @classmethod
    def _reject_user_written_auto(cls, data: Any) -> Any:
        # `from: auto` is not a value — recognition from the source is already
        # the default (an omitted `from:`). Catch the old spelling with a clear
        # message rather than letting it fail as an invalid description.
        if isinstance(data, dict) and data.get("from") == "auto":
            raise ValueError(
                "accumulate: 'from: auto' is not a value — describing the source data from "
                "the source is already the default, so omit 'from:' entirely"
            )
        return data

    @model_validator(mode="after")
    def _check(self) -> "AccumulateSchema":
        if not (isinstance(self.source, dict) and len(self.source) == 1):
            raise ValueError(f"accumulate: 'source' must have exactly one key, got {list(self.source.keys())}")

        # -- fold every spelling into `from:` ------------------------------
        self.from_, self.covering = normalise_from(
            from_=self.from_,
            accumulation=self.accumulation,
            covering=self.covering,
            availability=self.availability,
        )
        self.accumulation = None
        self.availability = None

        # Rules that need the output layout (whether a bare `from:` is a
        # valid-time source or a trajectory-layout scheme) live in the
        # `Recipe` model — the only place that knows the layout.

        if self.patch is not None:
            from ..field_to_interval import patch_registry

            for key in self.patch:
                if key not in patch_registry:
                    raise ValueError(f"accumulate: unknown patch {key!r} (expected one of {sorted(patch_registry)})")

        return self
