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

from anemoi.datasets import open_dataset

from ..tasks.gridded.tasks import FieldTask

LOG = logging.getLogger(__name__)


class Size(FieldTask):
    """A class to compute the size of a dataset."""

    def __init__(self, path: str, **kwargs: Any):
        """Initialize a Size instance.

        Parameters
        ----------
        path : str
            The path to the dataset.
        """
        super().__init__(path)

    def run(self) -> None:
        """Run the size computation."""
        from anemoi.datasets.create.size import compute_directory_sizes

        metadata = compute_directory_sizes(self.path)
        self.update_metadata(**metadata)

        # Look for constant fields
        ds = open_dataset(self.path)
        constants = ds.computed_constant_fields()

        variables_metadata = self.dataset.zarr_metadata.get("variables_metadata", {}).copy()
        for k in constants:
            variables_metadata[k]["constant_in_time"] = True

        self.update_metadata(constant_fields=constants, variables_metadata=variables_metadata)
