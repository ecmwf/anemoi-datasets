# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import textwrap
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

from anemoi.utils.humanize import plural

from .trace import step
from .trace import trace

LOG = logging.getLogger(__name__)


class Context:
    """Class to handle the build context in the dataset creation process."""

    def __init__(self) -> None:
        """Initializes a Context instance."""
        # used_references is a set of reference paths that will be needed
        self.used_references = set()
        # results is a dictionary of reference path -> obj
        self.results = {}

    def will_need_reference(self, key: Union[List, Tuple]) -> None:
        """Marks a reference as needed.

        Parameters
        ----------
        key : Union[List, Tuple]
            The reference key.
        """
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        self.used_references.add(key)

    def notify_result(self, key: Union[List, Tuple], result: Any) -> None:
        """Notifies that a result is available for a reference.

        Parameters
        ----------
        key : Union[List, Tuple]
            The reference key.
        result : Any
            The result object.
        """
        trace(
            "🎯",
            step(key),
            "notify result",
            textwrap.shorten(repr(result).replace(",", ", "), width=40),
            plural(len(result), "field"),
        )
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        if key in self.used_references:
            if key in self.results:
                raise ValueError(f"Duplicate result {key}")
            self.results[key] = result

    def get_result(self, key: Union[List, Tuple]) -> Any:
        """Retrieves the result for a given reference.

        Parameters
        ----------
        key : Union[List, Tuple]
            The reference key.

        Returns
        -------
        Any
            The result for the given reference.
        """
        assert isinstance(key, (list, tuple)), key
        key = tuple(key)
        if key in self.results:
            return self.results[key]
        all_keys = sorted(list(self.results.keys()))
        raise ValueError(f"Cannot find result {key} in {all_keys}")
