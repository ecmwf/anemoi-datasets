# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""One window of instantaneous samples collapsed to one field.

:class:`SampleReducer` is one object per ``(valid_date, basetime, grouping key)``,
holding the samples it still expects and the running reduction of those it has seen.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
from anemoi.transform.fields import Field
from anemoi.utils.dates import frequency_to_string
from numpy.typing import NDArray

from ..windowed.operations import Operation
from ..windowed.operations import stamp

LOG = logging.getLogger(__name__)


class SampleReducer:
    """Reduce the source samples of one window to a single field.

    Parameters
    ----------
    valid_date : datetime.datetime
        The validity time the reduced field is stamped with (the end of the window).
    period : datetime.timedelta
        The length of the window being reduced.
    key : tuple
        The grouping key (the field metadata that identifies the variable).
    samples : list of datetime.datetime
        The sample times the window is made of; all of them are required.
    operation : Operation
        How the samples are reduced.
    basetime : datetime.datetime, optional
        The model-run time to stamp the output with, for a trajectory row. ``None``
        (the gridded case) stamps the start of the window instead, so the whole step
        is the reduced window.
    """

    def __init__(
        self,
        valid_date: datetime.datetime,
        period: datetime.timedelta,
        key: tuple,
        samples: list[datetime.datetime],
        operation: Operation,
        basetime: datetime.datetime | None = None,
    ) -> None:
        self.valid_date = valid_date
        self.period = period
        self.key = key
        self.operation = operation
        self.basetime = basetime

        self.todo = list(samples)
        self.done: list[datetime.datetime] = []

        self.values: NDArray | None = None
        self.locked = False

    @property
    def outstanding(self) -> int:
        return len(self.todo)

    def is_complete(self) -> bool:
        """Check whether every sample of the window has been seen."""
        return not self.todo

    def compute(self, values: NDArray, valid_datetime: datetime.datetime) -> bool:
        """Reduce one source sample into the running value.

        Values are read from the field by the caller so that a field shared by several
        windows is decoded once.

        Returns
        -------
        bool
            ``True`` if the sample belonged to this window, ``False`` if it was not
            needed (another window will claim it).
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

        # A copy is mandatory: the same array is offered to every window that needs
        # this sample, and the operation may reduce into it.
        self.values = self.operation.reduce(self.values, values.copy(), 1.0)

        self.todo.remove(valid_datetime)
        self.done.append(valid_datetime)
        return True

    def as_field(self, template: Field) -> Field:
        """Build the reduced field once the window is complete."""
        assert self.is_complete(), (self.todo, self.done, self)
        assert not self.locked  # prevent building the field twice

        values = self.operation.finalize(self.values, float(len(self.done)))
        self.locked = True
        return stamp(values, template, self.operation, self.valid_date, self.period, self.basetime)

    def __repr__(self, verbose: bool = False) -> str:
        key = ", ".join(f"{k}={v}" for k, v in self.key)
        period = frequency_to_string(self.period)
        run = f", basetime={self.basetime}" if self.basetime is not None else ""
        default = (
            f"{type(self).__name__}(valid_date={self.valid_date}{run}, {period}, "
            f"{self.operation.name}, key={{ {key} }})"
        )
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
