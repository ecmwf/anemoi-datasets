# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import os
from typing import Any

import numpy as np

from ..creator import Creator
from ..dataset import Dataset
from ..statistics import StatisticsCollector
from .context import TabularContext

LOG = logging.getLogger(__name__)


class TabularCreator(Creator):

    allow_nans = True

    ######################################################

    def check_dataset_name(self, path: str) -> None:
        """Check the dataset name for validity.

        Parameters
        ----------
        path : str
            The path to the dataset to be checked.
        """
        pass

    def collect_metadata(self, metadata: dict) -> None:
        # See if that can be combined with `gridded`

        variables = self.minimal_input.variables
        LOG.info(f"Found {len(variables)} variables : {', '.join(variables)}.")
        metadata["variables"] = variables

    def initialise_dataset(self, dataset: Dataset) -> None:
        """Initialise the dataset arrays and coordinates for tabular data.

        Parameters
        ----------
        dataset : Dataset
            The dataset object to be initialised with arrays and coordinates.
        """
        pass

    ######################################################

    def context(self) -> TabularContext:
        return TabularContext(self.recipe)

    def load_result(self, result: Any, dataset: Dataset) -> None:
        """Load the result into the dataset by saving it as a NumPy file.

        Parameters
        ----------
        result : Any
            The result object containing the data to be loaded.
        dataset : Dataset
            The dataset object into which the result will be loaded.
        """
        os.makedirs(self.work_dir, exist_ok=True)

        # Split large arrays into multiple files so that the finalisation step
        # does not need to load huge files into memory.

        # TODO: read value from recipe

        max_size = 256 * 1024 * 1024  # 256 MB
        array = result.to_numpy()
        if array.shape[0] == 0:
            np.save(os.path.join(self.work_dir, f"{result.start_range}-{result.end_range}.npy"), array)
            return

        one_row_size = array.shape[1] * array.itemsize
        rows_per_file = max(round(max_size / one_row_size), 1)

        for i, row_start in enumerate(range(0, array.shape[0], rows_per_file)):
            row_end = min(row_start + rows_per_file, array.shape[0])

            np.save(
                os.path.join(self.work_dir, f"{result.start_range}-{result.end_range}-{i:04d}.npy"),
                array[row_start:row_end],
            )

    def finalise_dataset(self, dataset: Dataset) -> None:
        """Finalise the dataset after all data has been loaded.

        Parameters
        ----------
        dataset : Dataset
            The dataset object to be finalised.
        """
        from .finalise import finalise_tabular_dataset

        collector = StatisticsCollector(variables_names=self.variables_names)

        finalise_tabular_dataset(
            store=dataset.store,
            work_dir=self.work_dir,
            date_indexing=self.recipe.output.date_indexing,
            statistic_collector=collector,
            delete_files=False,
        )

        for name, data in collector.statistics().items():
            # Ignore date, time, latitude and longitude
            dataset.add_array(name=name, data=data[4:], dimensions=("variable",))

    def compute_and_store_statistics(self, dataset: Dataset) -> None:
        """Compute and store statistics for the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset object for which statistics will be computed and stored.
        """
        # TODO: implement if needed to recompute statistics
        raise NotImplementedError("Statistics are computed during finalisation for tabular datasets.")
