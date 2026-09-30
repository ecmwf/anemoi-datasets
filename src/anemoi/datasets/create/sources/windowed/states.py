# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""What a window is made of, while it is being filled.

A window is described by **subwindows** (spans of time, each rebuilt from one or more
archived fields) or by **samples** (instants, one field each). Those descriptions are
frozen and computed before any data arrives.

The classes here are their *mutable counterparts*: a :class:`State` gathers the fields
one part of a window needs and yields the value they make, plus the weight that part
carries. It is runtime state, not a fifth kind of time span --

    Subwindow      frozen:  "[06:00,07:00] is +a(0,7) -a(0,6)"
    SubwindowState mutable: "still waiting on a(0,7); partial sum so far is ..."

The point of the abstraction is that :class:`~.reducer.Reducer` holds both kinds
uniformly, so there is one reducer rather than two. A :class:`SubwindowState` sums its
contributions with their signs -- this is where differencing happens, and it may take
several fields. A :class:`SampleState` takes one field as it is.

The reducer therefore never sees a sign and never sees an archived interval: the two
levels stay apart.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from anemoi.datasets.create.intervals import SignedInterval

from .samples import Sample
from .subwindows import Subwindow

#: Statistics a field may carry that are *not* additive, so subtracting two of them
#: does not give the statistic over the difference of their intervals. `instant` and an
#: unknown value are deliberately absent: GRIB1 stores accumulations with
#: ``stepType=instant`` and says nothing, so treating that as a failure would reject
#: most of the archives `accumulate` exists to read.
NON_ADDITIVE = frozenset({"max", "min", "avg"})


def field_statistic(field: Any) -> str:
    """What statistic *field* carries over its interval, if it says.

    Two sources, because GRIB1 cannot state it in the message: ``proc.time_method`` is
    earthkit's reading of the message, right on GRIB2 and ``instant`` on GRIB1
    whatever the field really is; ``stepTypeForConversion`` is derived from the
    *parameter* (``tp`` is an accumulation, ``10fg`` a maximum), which is where GRIB1
    keeps it.

    Returns
    -------
    str
        ``accum``, ``max``, ``min``, ``avg``, or ``instant`` when neither source says
        anything useful. Never ``None``: "the field does not say" and "the field says
        instant" are the same answer here, because GRIB1 spells the first as the
        second, and both leave the field permitted.
    """
    for read in (lambda: field.get("proc.time_method"), lambda: field.metadata("stepTypeForConversion")):
        try:
            value = read()
        except Exception:  # pragma: no cover - backend-dependent
            continue
        value = str(value) if value is not None else None
        # `instant` from the message is not an answer on GRIB1; ask the parameter too.
        if value not in (None, "instant", "unknown"):
            return value
    return "instant"


class State(ABC):
    """One part of a window, and the value it contributes."""

    #: How much this state counts for in a weighted reduction.
    weight: float

    #: Whether this state rebuilds its part by *subtracting* archived fields. A field
    #: carrying a statistic in :data:`NON_ADDITIVE` may not fill such a state -- see
    #: :meth:`why_not`. Declared by the state because only the state knows how its
    #: part is put together; :class:`~.reducer.Reducer` must not have to ask what
    #: kind of state it is holding.
    differences: bool = False

    @property
    @abstractmethod
    def is_complete(self) -> bool:
        """Whether every field this state needs has arrived."""

    @abstractmethod
    def wants(self, identity: Any) -> bool:
        """Whether this state is still waiting for a field with this identity.

        The question :meth:`accept` answers on the way to taking the values, asked
        on its own so a caller can test it without offering anything.
        """

    @abstractmethod
    def accept(self, values: NDArray, identity: Any) -> bool:
        """Take a field if this state needs it; True if it did."""

    @abstractmethod
    def release(self) -> NDArray:
        """Hand over the state's value and drop the reference."""

    def why_not(self, statistic: str) -> str:
        """Why a field carrying *statistic* must not fill this state.

        Only called when :attr:`differences` is true, so the base implementation is
        never reached; a state that sets the flag owns the explanation.
        """
        raise NotImplementedError


class SubwindowState(State):
    """A subwindow, rebuilt from its contributions by a signed sum.

    Parameters
    ----------
    subwindow : Subwindow
        The part and the archived intervals that reconstruct it.
    """

    def __init__(self, subwindow: Subwindow) -> None:
        self.subwindow = subwindow
        self.weight = subwindow.weight
        # A direct subwindow is one archived field taken as it is; anything else is
        # rebuilt by a signed sum, which only means something for an additive quantity.
        self.differences = not subwindow.is_direct
        self.todo: list[SignedInterval] = list(subwindow.contributions)
        self.done: list[SignedInterval] = []
        self._values: NDArray | None = None

    @property
    def is_complete(self) -> bool:
        return not self.todo

    def _match(self, interval: SignedInterval) -> SignedInterval | None:
        for candidate in self.todo:
            if candidate.min == interval.min and candidate.max == interval.max and candidate.base == interval.base:
                return candidate
            if candidate.start == interval.start and candidate.end == interval.end and candidate.base is None:
                return candidate
        return None

    def wants(self, identity: Any) -> bool:
        return self._match(identity) is not None

    def why_not(self, statistic: str) -> str:
        return (
            f"field carrying a {statistic!r} is used to reconstruct {self.subwindow} by "
            f"differencing {len(self.subwindow.contributions)} archived fields. A "
            f"{statistic!r} is not additive, so the result would be meaningless. Either "
            "this parameter is stored per step and 'from:' describes a different layout, "
            "or this window cannot be built from this archive at all."
        )

    def accept(self, values: NDArray, identity: Any) -> bool:
        matching = self._match(identity)
        if matching is None:
            return False

        assert isinstance(values, np.ndarray), type(values)
        # `values` is shared with every other window this field feeds, so copy before
        # taking ownership: the reduction may consume what we release.
        local = matching.sign * values.copy()
        if self._values is None:
            self._values = local
        else:
            self._values += local

        self.todo.remove(matching)
        self.done.append(matching)
        return True

    def release(self) -> NDArray:
        values, self._values = self._values, None
        return values

    def __repr__(self) -> str:
        state = "complete" if self.is_complete else f"{len(self.todo)} outstanding"
        return f"{self.subwindow} [{state}]"


class SampleState(State):
    """One instantaneous field, taken as it is.

    Parameters
    ----------
    sample : Sample
        The instant this state expects a field for.
    """

    def __init__(self, sample: Sample) -> None:
        self.sample = sample
        self.weight = sample.weight
        self._values: NDArray | None = None
        self._seen = False

    @property
    def is_complete(self) -> bool:
        return self._seen

    def wants(self, identity: Any) -> bool:
        return not self._seen and identity[0] == self.sample.valid_datetime

    def accept(self, values: NDArray, identity: Any) -> bool:
        if not self.wants(identity):
            return False
        assert isinstance(values, np.ndarray), type(values)
        # shared between every window that needs this sample
        self._values = values.copy()
        self._seen = True
        return True

    def release(self) -> NDArray:
        values, self._values = self._values, None
        return values

    def __repr__(self) -> str:
        return f"{self.sample} [{'complete' if self._seen else 'outstanding'}]"
