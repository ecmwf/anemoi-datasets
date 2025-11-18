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


class Verify(FieldTask):
    """A class to verify the integrity of a dataset."""

    def __init__(self, path: str, **kwargs: Any):
        """Initialize a Verify instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        """
        super().__init__(path)

    def run(self) -> None:
        """Run the verification."""
        LOG.info(f"Verifying dataset at {self.path}")
        LOG.info(str(self.dataset.anemoi_dataset))
