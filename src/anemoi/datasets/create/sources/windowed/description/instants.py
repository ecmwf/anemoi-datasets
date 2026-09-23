# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""``from:`` — the source-data description for the time-reduction sources.

``period:`` says what the user wants; ``from:`` says what the source data
is.  This mirrors ``accumulate``, with two shapes, recognised structurally by
whether ``base_dates`` is present:

- :class:`FromInstants`, ``from: {frequency: 6h}`` — **base-less** source
  data: a regular grid of *instantaneous* fields, one every ``frequency``,
  indexed by validity time alone (analyses, reanalyses).  Usable under either
  output layout.
- :class:`FromRun`, ``from: {base_dates: true, frequency: 1h}`` — the
  source data is the forecast **run the trajectory layout imposes**, whose
  instantaneous fields exist every ``frequency`` of lead time.  Only
  meaningful under ``layout: trajectories``.  ``accumulate``'s
  ``base_dates: from-layout`` spelling is accepted too, silently.

The contrast with ``accumulate``'s bare ``from: {accumulation: 6h}`` is the
whole distinction the two keys carry:

- ``accumulation: 6h`` — each field **spans** a 6 h interval;
- ``frequency: 6h`` — fields **exist every** 6 h, each one an instant.

:class:`FromRun` deliberately takes **no** ``steps:``.  The lead times to
fetch are *derived* from the output steps and ``period`` — for each output
step ``s`` the window ``(s − period, s]`` needs the lead times
``s − period + k·frequency`` — so they are generally denser than the output
steps and reach below them.  Declaring ``steps:`` would state something the
source can work out, and ``steps: from-layout`` would actively mislead by
suggesting the samples sit on the output grid.

Unlike ``accumulate``, ``from:`` is **required**: it cannot be recognised
from a well-known MARS archive.  The recognition table
(``accumulate/description/mars_archives.py``) is keyed on ``(class, stream,
origin)`` alone and is param-blind — it answers ``from-zero`` for ``od-oper``
whether the request is for ``tp`` or ``2t``.  For a sum that is harmless, but
reducing an instantaneous parameter as if it were accumulated silently
produces differences of temperatures rather than temperatures, so the
description has to be stated.
"""

from __future__ import annotations

import datetime
from typing import Annotated
from typing import Any

from anemoi.utils.dates import frequency_to_string
from pydantic import BaseModel
from pydantic import BeforeValidator
from pydantic import ConfigDict
from pydantic import Field
from pydantic import TypeAdapter
from pydantic import model_validator

from anemoi.datasets.create.time_schemas import Frequency

#: Keys belonging to ``accumulate``'s ``from:`` vocabulary.  They describe
#: interval-valued source data, so they are meaningless here and are worth a
#: better message than pydantic's "extra inputs are not permitted".
#: ``base_dates`` is *not* in the list: it is this package's discriminator
#: between the base-less and the run-anchored shape.
_ACCUMULATE_ONLY_KEYS = (
    "accumulation",
    "lookup_table",
    "lookup-table",
)

#: ``accumulate``'s spelling of the same idea.  Accepted on input and folded
#: into ``True``; not documented, because there is no paired ``steps:
#: from-layout`` here for it to agree with, so it is a bare flag and ``true``
#: says that better.
FROM_LAYOUT = "from-layout"


def _coerce_base_dates(value: Any) -> Any:
    """Fold ``accumulate``'s ``from-layout`` spelling into the ``True`` flag."""
    return True if value == FROM_LAYOUT else value


def _hyphen_alias(name: str) -> str:
    return name.replace("_", "-")


class FromInstants(BaseModel):
    """``from: {frequency: ...}`` — instantaneous fields on a regular grid.

    Parameters
    ----------
    frequency : datetime.timedelta
        The spacing of the source fields in time, e.g. ``6h``.  This is the
        cadence of the *source data*, which is generally not the ``dates``
        frequency of the dataset being built.
    """

    model_config = ConfigDict(extra="forbid")

    frequency: Frequency

    @model_validator(mode="after")
    def _check(self) -> "FromInstants":
        _check_frequency(self.frequency)
        return self


class FromRun(BaseModel):
    """``from: {base_dates: true, frequency: ...}`` — the layout's own run.

    The source data is the forecast run the trajectory layout initialises, and
    its instantaneous fields exist every ``frequency`` of lead time.  The lead
    times actually fetched are derived from the output steps and ``period``
    (see the module docstring), so there is no ``steps:`` key.

    Parameters
    ----------
    base_dates : bool
        ``true``: the run is the one the output layout imposes.  It is a plain
        flag rather than a sentinel string because, unlike ``accumulate``,
        there is no paired ``steps:`` for it to agree with.  An explicit table
        of base dates — a *different* forecast archive — is not supported yet.
    frequency : datetime.timedelta
        The spacing of the source fields in lead time, e.g. ``1h``.  This is
        the cadence of the *source run*, which is generally denser than the
        output ``steps.frequency``.
    """

    model_config = ConfigDict(extra="forbid")

    base_dates: Annotated[bool, BeforeValidator(_coerce_base_dates)]
    frequency: Frequency

    @model_validator(mode="after")
    def _check(self) -> "FromRun":
        if self.base_dates is not True:
            raise ValueError(
                "'from.base_dates: false' is not a value — omit 'base_dates' entirely to read "
                "base-less source data, i.e. 'from: {frequency: ...}'"
            )
        _check_frequency(self.frequency)
        return self


#: Recognised structurally: ``base_dates`` present means run-anchored.
From = FromRun | FromInstants


def _check_frequency(frequency: datetime.timedelta) -> None:
    """Check that a source cadence is a positive duration."""
    if frequency <= datetime.timedelta(0):
        raise ValueError(f"'from.frequency' must be positive, got {frequency_to_string(frequency)}")


def _precheck_from(value: Any) -> Any:
    """Reject the shapes that would otherwise fail as an opaque union error.

    A pydantic union reports "Field required" for every member it tried, which
    tells a recipe author nothing.  These three cases are the ones worth
    naming, so they are caught before the union is attempted.
    """
    if not isinstance(value, dict):
        return value

    found = sorted(k for k in value if k in _ACCUMULATE_ONLY_KEYS)
    if found:
        raise ValueError(
            f"'from:' does not accept {found} — those describe interval-valued source data "
            "and belong to 'accumulate:'. A time reduction reads instantaneous fields; "
            "state their cadence with 'from: {frequency: ...}'"
        )

    base_dates = value.get("base_dates", value.get("base-dates"))
    if base_dates is not None and base_dates is not True and base_dates != FROM_LAYOUT:
        if base_dates is False:
            raise ValueError(
                "'from.base_dates: false' is not a value — omit 'base_dates' entirely to read "
                "base-less source data, i.e. 'from: {frequency: ...}'"
            )
        raise ValueError(
            "'from.base_dates' only accepts 'true' — reducing instantaneous fields from a "
            "*different* forecast archive (an explicit base_dates table) is not supported "
            "yet; it needs run selection. Use 'base_dates: true' for the run the trajectory "
            "layout imposes, or a base-less 'from: {frequency: ...}'"
        )

    if "steps" in value:
        raise ValueError(
            "'from:' does not take 'steps' — the sample lead times are derived from the "
            "output 'steps' and 'period' (the window (s - period, s] needs the lead times "
            "s - period + k*frequency), so they are denser than the output steps. State "
            "only the cadence, with 'frequency:'"
        )

    if "frequency" not in value:
        raise ValueError(
            "'from:' needs a 'frequency:' — the cadence of the source data, e.g. " "'from: {frequency: 6h}'"
        )

    return value


#: The ``from:`` field as it appears in the schema: pre-checked, then matched
#: structurally against the union.
FromField = Annotated[From, BeforeValidator(_precheck_from)]

#: Built once — a per-call model would rebuild pydantic's validator every time.
_FROM_ADAPTER: TypeAdapter[FromField] = TypeAdapter(FromField)


def validate_from(value: Any) -> From:
    """Validate a ``from:`` payload into one of the descriptions.

    Shared by :class:`ReduceSchema` (recipe time) and ``ReduceSource`` (build
    time) so the two cannot drift apart.

    Parameters
    ----------
    value : Any
        The ``from:`` block, as a mapping or an already-validated member.

    Returns
    -------
    From
        The validated description.
    """
    if isinstance(value, (FromInstants, FromRun)):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"'from:' must be a mapping, got {value!r}")
    return _FROM_ADAPTER.validate_python(value)


class ReduceSchema(BaseModel):
    """Validation schema for the ``average``, ``minimum`` and ``maximum`` sources.

    The three keys mirror ``accumulate``: ``source:`` is where the data comes
    from, ``period:`` is the window you want, and ``from:`` is what the source
    data is.  Unlike ``accumulate``, ``from:`` is required (see the module
    docstring).

    Whether a run-anchored ``from:`` is allowed depends on the output layout,
    so that rule lives in the ``Recipe`` model — the only place that knows the
    layout — exactly as it does for ``accumulate``.
    """

    model_config = ConfigDict(
        alias_generator=_hyphen_alias,
        populate_by_name=True,
        extra="forbid",
    )

    period: Frequency
    source: dict[str, Any]
    from_: FromField = Field(alias="from")

    group_by: dict[str, Any] | None = None

    @property
    def is_run_anchored(self) -> bool:
        """Whether ``from:`` describes the run the trajectory layout imposes."""
        return isinstance(self.from_, FromRun)

    @model_validator(mode="before")
    @classmethod
    def _require_from(cls, data: Any) -> Any:
        # `from:` cannot be recognised from the source (see the module
        # docstring), so say that rather than leaving pydantic to report a
        # bare "Field required".
        if isinstance(data, dict) and data.get("from", data.get("from_")) is None:
            raise ValueError(
                "'from:' is required — state the cadence of the source data, e.g. "
                "'from: {frequency: 6h}'. It cannot be recognised from the source: the "
                "archive table is param-blind, and reducing an instantaneous parameter as "
                "if it were accumulated fails silently"
            )
        return data

    @model_validator(mode="after")
    def _check(self) -> "ReduceSchema":
        if not (isinstance(self.source, dict) and len(self.source) == 1):
            raise ValueError(f"'source' must have exactly one key, got {sorted(self.source)}")

        check_period(self.period, self.from_.frequency)
        return self


def check_period(period: datetime.timedelta, frequency: datetime.timedelta) -> None:
    """Check that a reduction window is a whole number of source samples.

    Parameters
    ----------
    period : datetime.timedelta
        The requested reduction window.
    frequency : datetime.timedelta
        The cadence of the source data (``from.frequency``).
    """
    if period <= datetime.timedelta(0):
        raise ValueError(f"'period' must be positive, got {frequency_to_string(period)}")

    if period % frequency != datetime.timedelta(0):
        raise ValueError(
            f"'from.frequency' ({frequency_to_string(frequency)}) must divide the requested "
            f"'period' ({frequency_to_string(period)}) — the window is reduced over whole "
            "source samples"
        )


def window_samples(
    valid_date: datetime.datetime,
    period: datetime.timedelta,
    frequency: datetime.timedelta,
) -> list[datetime.datetime]:
    """The source sample times reduced into the field stamped at *valid_date*.

    The window is end-anchored and half-open, ``(valid_date − period,
    valid_date]``, which is the convention an anemoi dataset uses throughout
    and the one ``accumulate`` reconstructs for a sum.  A 24 h window over
    6-hourly source data is therefore the four samples at ``−18h``, ``−12h``,
    ``−6h`` and ``0h`` — the start of the window belongs to the previous one.

    Parameters
    ----------
    valid_date : datetime.datetime
        The validity time the reduced field is stamped with.
    period : datetime.timedelta
        The reduction window.
    frequency : datetime.timedelta
        The cadence of the source data.

    Returns
    -------
    list of datetime.datetime
        The sample times, in ascending order, ending at *valid_date*.
    """
    check_period(period, frequency)
    count = period // frequency
    start = valid_date - period
    return [start + (i + 1) * frequency for i in range(count)]


def check_window_inside_run(
    valid_date: datetime.datetime,
    basetime: datetime.datetime,
    period: datetime.timedelta,
    source_name: str = "reduce",
) -> None:
    """Check that a run-anchored window does not straddle the basetime.

    The samples come from the run initialised at *basetime*, so the whole
    window ``(valid_date − period, valid_date]`` has to lie at or after it.  A
    window reaching further back would need fields from before the forecast —
    which is analysis, a different quantity, not merely missing data.

    This rule applies only to a run-anchored ``from:``.  A base-less ``from:``
    reads an analysis archive, where data before the basetime exists and is
    the same quantity, so no such restriction holds there.

    Parameters
    ----------
    valid_date : datetime.datetime
        The validity time of the output row (``basetime + step``).
    basetime : datetime.datetime
        The run's initialisation time.
    period : datetime.timedelta
        The reduction window.
    source_name : str
        Names the recipe block in the error message.
    """
    step = valid_date - basetime
    if step < period:
        raise ValueError(
            f"{source_name}: the window of {valid_date} straddles the basetime {basetime} — "
            f"step {frequency_to_string(step)} is shorter than 'period' "
            f"({frequency_to_string(period)}), so the window would need fields from before "
            "the forecast. Start 'steps' at or after 'period', or read a base-less source "
            "with 'from: {frequency: ...}'"
        )
