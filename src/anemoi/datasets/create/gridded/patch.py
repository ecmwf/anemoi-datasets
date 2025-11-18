# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

from ..gridded.tasks import FieldTask

LOG = logging.getLogger(__name__)


class Patch(FieldTask):
    """A class to apply patches to a dataset."""

    def __init__(self, path: str, options: dict = None, **kwargs: Any):
        """Initialize a Patch instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        options : dict, optional
            The patch options.
        """
        self.path = path
        self.options = options or {}

    def run(self) -> None:
        """Run the patch."""
        from anemoi.datasets.create.patch import apply_patch

        apply_patch(self.path, **self.options)
