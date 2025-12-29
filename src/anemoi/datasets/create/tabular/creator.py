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
        pass

    def collect_metadata(self, metadata: dict):
        # See if that can be combined with `gridded`

        variables = self.minimal_input.variables
        LOG.info(f"Found {len(variables)} variables : {','.join(variables)}.")
        metadata["variables"] = variables

    def initialise_dataset(self, dataset: Dataset) -> None:
        pass

    ######################################################

    def context(self):
        return TabularContext(self.recipe)

    def load_result(self, result: Any, dataset: Dataset) -> None:
        os.makedirs(self.work_dir, exist_ok=True)
        np.save(
            os.path.join(
                self.work_dir,
                f"{result.start_date}-{result.end_date}.npy",
            ),
            result.to_numpy(),
        )

    def finalise_dataset(self, dataset: Dataset) -> None:
        from .finalise import finalise_tabular_dataset

        collector = StatisticsCollector(variables_names=self.variables_names)

        finalise_tabular_dataset(
            store=dataset.store,
            work_dir=self.work_dir,
            date_indexing=self.recipe.date_indexing,
            statistic_collector=collector,
            delete_files=True,
        )

        for name, data in collector.statistics().items():
            dataset.add_array(name=name, data=data, dimensions=("variable",))

    def compute_and_store_statistics(self, dataset: Dataset) -> None:
        raise NotImplementedError("Statistics are computed during finalisation for tabular datasets.")
