# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``clip:`` — bounding the accumulated values.

The accumulated value of a physically non-negative quantity (precipitation,
radiation, …) can come out slightly negative, because it is built by
*differencing* cumulative fields and the archive holds them at finite
precision.  ``clip:`` bounds the result, exactly as the ``clip`` filter of
*anemoi-transform* piped after the source would:

.. code:: yaml

   - accumulate:
       period: 6h
       source: {...}
       clip: {param: tp, minimum: 0}

is the same computation as

.. code:: yaml

   - pipe:
       - accumulate: {period: 6h, source: {...}}
       - clip: {param: tp, minimum: 0}

and it *is* the same code: :func:`apply_clip` builds the transform filter
and lets it do the arithmetic, so the two cannot drift apart.  The
configuration shape is the filter's own — ``param``, ``minimum``,
``maximum``, at least one bound — with two differences that the inline
form buys:

- ``param`` may be omitted, meaning *every accumulated parameter* (the
  filter itself requires one, so the omitted form is expanded here into
  one filter per accumulated parameter);
- a ``param`` that does not name an accumulated parameter is an error,
  where the filter would silently pass every field through.

The clipping applies to the **accumulated** value, not to the intervals it
was built from: in a 6h accumulation assembled from 3h differences, a
negative increment cancelled by a positive one is never seen here.
"""

from __future__ import annotations

import logging
from typing import Any

from anemoi.transform import FieldList
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import model_validator

LOG = logging.getLogger(__name__)

# The component path the `clip` filter selects fields on.
VARIABLE = "parameter.variable"


class ClipSpec(BaseModel):
    """One ``clip:`` entry, with the same shape as the ``clip`` filter.

    ``param`` is optional here (the filter requires it); when omitted the
    entry applies to every accumulated parameter.
    """

    model_config = ConfigDict(extra="forbid")

    param: str | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def _check(self) -> "ClipSpec":
        # Same rule as the filter's `prepare_filter`: no bound is inferred.
        if self.minimum is None and self.maximum is None:
            raise ValueError(
                "accumulate clip: at least one value for 'minimum' or 'maximum' must be specified "
                "(e.g. 'clip: {param: tp, minimum: 0}')"
            )
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"accumulate clip: minimum ({self.minimum}) is greater than maximum ({self.maximum})")
        return self

    def filter_config(self, param: str) -> dict[str, Any]:
        """The ``clip`` filter configuration for one parameter."""
        return {"param": param, "minimum": self.minimum, "maximum": self.maximum}

    def __str__(self) -> str:
        param = self.param if self.param is not None else "(all)"
        return f"{param} → [{self.minimum}, {self.maximum}]"


def normalise_clip(clip: Any) -> list[ClipSpec] | None:
    """Fold every accepted spelling of ``clip:`` into a list of :class:`ClipSpec`.

    Shared by recipe-time validation (``AccumulateSchema``) and the runtime
    source, so the two cannot drift apart.

    Parameters
    ----------
    clip : Any
        The recipe value: ``None`` or ``False`` (no clipping), one mapping,
        or a list of mappings. Already-built :class:`ClipSpec` instances are
        accepted too (a schema that has been dumped and rebuilt).

    Returns
    -------
    list of ClipSpec or None
        ``None`` when no clipping is requested.
    """
    if clip is None or clip is False:
        return None

    if clip is True:
        raise ValueError(
            "accumulate: 'clip: true' is not a value — give the bounds explicitly, "
            "as the 'clip' filter does, e.g. 'clip: {param: tp, minimum: 0}'"
        )

    if isinstance(clip, (ClipSpec, dict)):
        clip = [clip]

    if not isinstance(clip, (list, tuple)):
        raise ValueError(f"accumulate: 'clip' must be a mapping or a list of mappings, got {clip!r}")

    specs: list[ClipSpec] = []
    for item in clip:
        if isinstance(item, ClipSpec):
            specs.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(
                f"accumulate: each 'clip' entry must be a mapping with the same keys as the "
                f"'clip' filter (param, minimum, maximum), got {item!r}"
            )
        specs.append(ClipSpec(**item))

    if not specs:
        return None

    seen = []
    for spec in specs:
        if spec.param in seen:
            name = repr(spec.param) if spec.param is not None else "the omitted 'param'"
            raise ValueError(f"accumulate: 'clip' has more than one entry for {name}")
        seen.append(spec.param)

    return specs


def _variables(fields: FieldList) -> list[str]:
    """The distinct parameters present in *fields*, in order of first appearance."""
    result: list[str] = []
    for field in fields:
        variable = field.get(VARIABLE, None)
        if variable is not None and variable not in result:
            result.append(variable)
    return result


def apply_clip(fields: FieldList, specs: list[ClipSpec] | None) -> FieldList:
    """Clip the accumulated fields, delegating the arithmetic to the ``clip`` filter.

    Parameters
    ----------
    fields : FieldList
        The accumulated fields.
    specs : list of ClipSpec or None
        The normalised ``clip:`` entries; ``None`` returns *fields* unchanged.

    Returns
    -------
    FieldList
        The clipped fields.
    """
    if not specs:
        return fields

    from anemoi.transform.filters import create_filter_by_name

    available = _variables(fields)

    for spec in specs:
        if spec.param is None:
            params = available
        elif spec.param in available:
            params = [spec.param]
        else:
            raise ValueError(
                f"accumulate clip: parameter {spec.param!r} is not among the accumulated " f"parameters {available}"
            )

        for param in params:
            LOG.info("Clipping accumulated %s to [%s, %s]", param, spec.minimum, spec.maximum)
            fields = create_filter_by_name("clip", **spec.filter_config(param)).forward(fields)

    return fields
