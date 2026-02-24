# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import re

from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOG = logging.getLogger(__name__)


class Window:
    """Represents a time window for selecting data, with before/after offsets and inclusivity.

    Parses a window string to determine the time offsets before and after a central point,
    and whether the window is inclusive or exclusive at each end. Used by WindowView to select data slices.
    """

    def __init__(self, window: str) -> None:
        """Parse the window string and initialise the window parameters.

        Parameters
        ----------
        window : str
            String representation of the window, e.g. "(-3,+0]".
        """
        # Parse the window string using regex to extract bounds and inclusivity
        m = re.match(r"([\[\(])(.*),(.*)([\]\)])", window)
        if not m:
            raise ValueError(f"Window: invalid window string: {window}")
        # Convert before/after offsets to time_deltas
        self.before = frequency_to_timedelta(m.group(2))
        self.after = frequency_to_timedelta(m.group(3))
        # Determine if window is exclusive at each end
        self.exclude_before = m.group(1) == "("
        self.exclude_after = m.group(4) == ")"

    def __repr__(self) -> str:
        """Return a string representation of the window.

        Returns
        -------
        str
            The string representation of the window.
        """
        B = {True: ("(", ")"), False: ("[", "]")}
        return (
            f"{B[self.exclude_before][0]}{frequency_to_string(self.before)},"
            f"{frequency_to_string(self.after)}{B[self.exclude_after][1]}"
        )
