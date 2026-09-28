# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The reduction strategies — one window of samples collapsed to one field.

:class:`Reducer` is to the time-reduction sources what ``Accumulator`` is to
``accumulate``: one object per ``(valid_date, grouping key)``, holding the
samples it still expects and the running reduction of those it has seen.  The
subclasses differ only in how a sample is folded in (:meth:`Reducer._combine`)
and in the ``proc.time_method`` they stamp on the output.
"""

from __future__ import annotations

import datetime
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
from anemoi.transform.fields import Field
from anemoi.utils.dates import frequency_to_string
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


def _register_min_time_method() -> None:
    """Teach earthkit-data the ``min`` time-processing method.

    earthkit-data ships ``accum``, ``avg``, ``instant`` and ``max`` but no
    ``min`` (still true in 1.2.2), so stamping ``proc.time_method="min"`` on a
    field raises ``Unsupported time method type: min``.  anemoi-transform is
    already ready for it — ``_STEP_TYPE_FOR_CONVERSION`` maps ``min`` to the
    ``minimum`` statistical process — so the gap is only in the whitelist that
    validates the value.

    Without this, ``minimum`` would have to leave its output unstamped and the
    field would claim to be instantaneous.  The registration is idempotent and
    disappears of its own accord once earthkit-data adds ``MIN``.
    """
    from earthkit.data.field.component import time_span

    if "min" not in time_span._TIME_METHODS:
        time_span._TIME_METHODS["min"] = time_span.TimeMethod("min")


_register_min_time_method()


class Reducer(ABC):
    """Reduce the source samples of one window to a single field.

    Parameters
    ----------
    valid_date : datetime.datetime
        The validity time the reduced field is stamped with (the end of the
        window).
    period : datetime.timedelta
        The length of the window being reduced.
    key : tuple
        The grouping key (the field metadata that identifies the variable).
    samples : list of datetime.datetime
        The sample times the window is made of; all of them are required.
    basetime : datetime.datetime, optional
        The model-run time to stamp the output with, for a trajectory row.
        ``None`` (the gridded case) stamps the start of the window instead, so
        the whole step is the reduced window.
    """

    #: The earthkit ``proc.time_method`` recorded on the output field.  The
    #: values match the GRIB step types anemoi-transform maps to a statistical
    #: process (``avg`` → ``average``, ``min`` → ``minimum``, ``max`` →
    #: ``maximum``).
    time_method: str

    def __init__(
        self,
        valid_date: datetime.datetime,
        period: datetime.timedelta,
        key: tuple,
        samples: list[datetime.datetime],
        basetime: datetime.datetime | None = None,
    ) -> None:
        self.valid_date = valid_date
        self.period = period
        self.key = key
        self.basetime = basetime

        self.todo = list(samples)
        self.done: list[datetime.datetime] = []

        self.values: NDArray | None = None
        self.locked = False

    def is_complete(self) -> bool:
        """Check whether every sample of the window has been seen."""
        return not self.todo

    def compute(self, values: NDArray, valid_datetime: datetime.datetime) -> bool:
        """Fold one source sample into the running reduction.

        Values are read from the field by the caller so that a field shared by
        several windows is decoded once.

        Parameters
        ----------
        values : numpy.ndarray
            The sample's values.
        valid_datetime : datetime.datetime
            The sample's validity time.

        Returns
        -------
        bool
            ``True`` if the sample belonged to this window, ``False`` if it was
            not needed (another window will claim it).
        """
        if valid_datetime not in self.todo:
            if valid_datetime in self.done:
                raise ValueError(
                    f"Sample {valid_datetime} was already reduced into {self!r}; "
                    "the source returned the same field twice"
                )
            return False

        if self.locked:
            raise ValueError(f"{self!r} has already produced its field, cannot reduce {valid_datetime}")

        assert isinstance(values, np.ndarray), type(values)

        if self.values is None:
            # A copy is mandatory: the same array is offered to every window
            # that needs this sample.
            self.values = values.copy()
        else:
            self._combine(values)

        self.todo.remove(valid_datetime)
        self.done.append(valid_datetime)
        return True

    @abstractmethod
    def _combine(self, values: NDArray) -> None:
        """Fold *values* into ``self.values`` (never called for the first sample)."""

    def _result(self) -> NDArray:
        """The reduced values, once every sample has been folded in."""
        return self.values

    def as_field(self, template: Field) -> Field:
        """Build the reduced field once the window is complete.

        The result is an in-memory field stamped like an ``accumulate`` output.
        For a gridded reduction (no basetime) the base time is the start of the
        window and the step reaches the validity time, so the whole step is the
        reduced window.  For a trajectory row the base time is the model-run
        basetime, so trajectory loaders recover ``(basetime, step)``.  Either
        way the processing component records which reduction produced it, over
        ``period``.

        Parameters
        ----------
        template : Field
            Field providing all other components (parameter, geography, ...).

        Returns
        -------
        Field
            The reduced field.
        """
        assert self.is_complete(), (self.todo, self.done, self)
        assert not self.locked  # prevent building the field twice

        basetime = self.basetime if self.basetime is not None else self.valid_date - self.period
        field = Field.from_numpy(
            self._result(),
            template=template,
            **{
                "time.base_datetime": basetime,
                "time.step": self.valid_date - basetime,
                "proc.time_method": self.time_method,
                "proc.time_value": self.period,
            },
        )
        self.locked = True
        return field

    def __repr__(self, verbose: bool = False) -> str:
        key = ", ".join(f"{k}={v}" for k, v in self.key)
        period = frequency_to_string(self.period)
        run = f", basetime={self.basetime}" if self.basetime is not None else ""
        default = f"{type(self).__name__}(valid_date={self.valid_date}{run}, {period}, key={{ {key} }})"
        if verbose:
            extra = []
            if self.locked:
                extra.append("(locked)")
            for d in self.done:
                extra.append(f"    done: {d}")
            for d in self.todo:
                extra.append(f"    todo: {d}")
            default += "\n" + "\n".join(extra)
        return default


class AverageReducer(Reducer):
    """Arithmetic mean of the window's samples."""

    time_method = "avg"

    def _combine(self, values: NDArray) -> None:
        self.values += values

    def _result(self) -> NDArray:
        return self.values / len(self.done)


class MinimumReducer(Reducer):
    """Pointwise minimum of the window's samples."""

    time_method = "min"

    def _combine(self, values: NDArray) -> None:
        self.values = np.minimum(self.values, values)


class MaximumReducer(Reducer):
    """Pointwise maximum of the window's samples."""

    time_method = "max"

    def _combine(self, values: NDArray) -> None:
        self.values = np.maximum(self.values, values)


def describe(reducers: dict[Any, Reducer], limit: int = 20) -> str:
    """Render reducers for an error message, most incomplete first."""
    ordered = sorted(reducers.values(), key=lambda r: (-len(r.todo), r.valid_date))
    lines = [f"  {r.__repr__(verbose=True)}" for r in ordered[:limit]]
    if len(ordered) > limit:
        lines.append(f"  ... and {len(ordered) - limit} more")
    return "\n".join(lines)
