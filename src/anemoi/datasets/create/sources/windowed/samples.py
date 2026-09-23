# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A sample: one instant inside a window.

The other kind of part a window can be made of. Unlike a :class:`~.subwindows.Subwindow`
a sample is an *instant*, not a span, and samples therefore **do not partition** the
window: four 6-hourly samples of a 24h window cover nothing between them. That is why
``validate_partition`` never sees one, and why reducing samples is a different operation
from reducing a partition rather than a special case of it.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """One instantaneous field the window is reduced over.

    Parameters
    ----------
    valid_datetime : datetime.datetime
        The instant this sample is taken at.
    """

    valid_datetime: datetime.datetime

    @property
    def weight(self) -> float:
        """How much this part counts for in a weighted reduction.

        Samples sit on a regular cadence, so they weigh the same and a weighted mean
        over them is the plain arithmetic mean. A subwindow weighs its own length
        instead; keeping the weight on the *part* is what lets one ``Mean`` serve both.
        """
        return 1.0

    def __repr__(self) -> str:
        return f"Sample({self.valid_datetime:%Y%m%d.%H%M})"
